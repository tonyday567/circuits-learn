-- | Axioma oracle for the circuits-learn consolidation slice.
--
-- Verifies:
--   1. Para composition law (Category associativity + identity)
--   2. Para composition == Loop(,) constant-state slice
--   3. Pre-existing Adam / EWMA oracles
--
-- Non-zero exit code on any mismatch.
module Main where

import Circuit.Channel (trace)
import Circuit.Learn.Adam (adam, adamReference, adamUpdates, ewma)
import Circuit.Learn.Para (Para (..), forgetPara, liftPara, runPara)
import Circuit.Process (scan)
import Control.Category (id, (.))
import Data.List (scanl')
import System.Exit (exitFailure)
import Prelude hiding (id, (.))

-- | A short, deterministic gradient trace.
gradTrace :: [Double]
gradTrace = [0.5, -0.3, 0.2, -0.8, 0.1, 0.4, -0.2]

-- | Exact EWMA recurrence for comparison.
ewmaReference :: Double -> Double -> [Double] -> [Double]
ewmaReference _ _ [] = []
ewmaReference alpha s0 (x0 : xs) = scanl' step (alpha * x0 + (1 - alpha) * s0) xs
  where
    step s x = alpha * x + (1 - alpha) * s

-- | Tolerance for floating-point comparisons near zero.
epsTight :: Double
epsTight = 1e-12

epsLoose :: Double
epsLoose = 1e-10

near :: Double -> Double -> Double -> Bool
near tol a b = abs (a - b) <= tol

allNear :: Double -> [Double] -> [Double] -> Bool
allNear tol as bs = length as == length bs && and (zipWith (near tol) as bs)

report :: String -> Bool -> IO ()
report label ok = putStrLn $ label ++ ": " ++ if ok then "PASS" else "FAIL"

-- ---------------------------------------------------------------------------
-- Adam / EWMA (pre-existing)
-- ---------------------------------------------------------------------------

checkEWMA :: IO Bool
checkEWMA = do
  let alpha = 0.9
      s0 = 0.0
      expected = ewmaReference alpha s0 gradTrace
      actual = scan (ewma alpha s0) gradTrace
      ok = allNear epsTight expected actual
  report "EWMA vs hand recurrence" ok
  pure ok

checkAdam :: IO Bool
checkAdam = do
  let alpha = 0.01
      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8
      (g0, gs) = case gradTrace of (x : xs) -> (x, xs); [] -> error "empty trace"
      expected = adamUpdates alpha beta1 beta2 eps (adamReference beta1 beta2 g0 gs)
      actual = scan (adam alpha beta1 beta2 eps) gradTrace
      ok = allNear epsLoose expected actual
  report "Adam vs reference recurrence" ok
  pure ok

-- ---------------------------------------------------------------------------
-- Para oracles
-- ---------------------------------------------------------------------------

-- | Non-trivial 'Para' witnesses over 'Int'.
-- Each uses the parameter @p@ in a different way.
fInt, gInt, hInt :: Para Int Int Int
fInt = Para $ \(p, x) -> x + p
gInt = Para $ \(p, x) -> x * p
hInt = Para $ \(p, x) -> x - p

-- | Associativity: (h . g) . f == h . (g . f).
--
-- With @p=3, x=5@:
--   (hInt . gInt) . fInt: fInt(3,5)=8, gInt(3,8)=24, hInt(3,24)=21
--   hInt . (gInt . fInt): gInt(3,fInt(3,5))=gInt(3,8)=24, hInt(3,24)=21  ✓
checkAssoc :: IO Bool
checkAssoc = do
  let ps = [1, 2, 3, 5, 7, 10] :: [Int]
      xs = [0, 1, 5, 10, -3] :: [Int]
      leftAssoc = runPara (hInt . (gInt . fInt))
      rightAssoc = runPara ((hInt . gInt) . fInt)
      ok = and [leftAssoc p x == rightAssoc p x | p <- ps, x <- xs]
  report "Para associativity: (h . g) . f == h . (g . f)" ok
  pure ok

-- | Identity: @id . f == f@ and @f . id == f@.
checkIdentity :: IO Bool
checkIdentity = do
  let ps = [1, 2, 3, 5, 7, 10] :: [Int]
      xs = [0, 1, 5, 10, -3] :: [Int]
      ok1 = and [runPara (id . fInt) p x == runPara fInt p x | p <- ps, x <- xs]
      ok2 = and [runPara (fInt . id) p x == runPara fInt p x | p <- ps, x <- xs]
  report "Para identity: id . f == f" ok1
  report "Para identity: f . id == f" ok2
  pure (ok1 && ok2)

-- | Constant-state equivalence: para threading == trace.
--
-- For a function @f :: a -> b@ (ignoring the parameter), both paths give the
-- same result:
--   @forgetPara p (liftPara f) a == f a@  (by construction)
--   @trace (\\(s, a) -> (s, f a)) a == f a@  (constant-state trace law)
--
-- The trace law is L1/L2 from circuits/app/axioma.hs, promoted here.
checkConstantState :: IO Bool
checkConstantState = do
  -- L1: threading constant state through trace yields function application
  let l1Inputs = [42 :: Int, 0, -1, 100]
      f1 = show
      ok1 = and [trace (\(p, a) -> (p, f1 a)) x == f1 x | x <- l1Inputs]
  report "Trace constant-state: show (L1)" ok1
  -- L2: same with arithmetic
  let l2Inputs = [5 :: Int, 0, -3, 100]
      f2 = (+ 10)
      ok2 = and [trace (\(p, a) -> (p, f2 a)) x == f2 x | x <- l2Inputs]
  report "Trace constant-state: (+10) (L2)" ok2
  -- Para version: liftPara then forgetPara == identity on the function
  let ps = [1, 2, 3, 5] :: [Int]
      xs = [0, 1, 5, 10] :: [Int]
      ok3 = and [forgetPara p (liftPara (+ p)) x == x + p | p <- ps, x <- xs]
  report "Para: forgetPara p (liftPara f) == f" ok3
  pure (ok1 && ok2 && ok3)

main :: IO ()
main = do
  okEWMA <- checkEWMA
  okAdam <- checkAdam
  okAssoc <- checkAssoc
  okId <- checkIdentity
  okConstState <- checkConstantState
  let allOk = and [okEWMA, okAdam, okAssoc, okId, okConstState]
  if allOk
    then putStrLn "circuits-learn axioma: all checks passed"
    else do
      putStrLn "circuits-learn axioma: one or more checks failed"
      exitFailure
