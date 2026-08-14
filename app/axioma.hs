-- | Axioma oracle for the circuits-learn consolidation slice.
--
-- Verifies:
--   1. Para composition law (Category associativity + identity)
--   2. Para composition == Loop(,) constant-state slice
--   3. Adam / EWMA oracles
--   4. Two-EWMA Adam decomposition
--   5. Toy 2-layer NN fit (loss decreases)
--   6. Ephemeral 'Progress'/'Experience' framing: SGD over a dataset decreases
--      loss.
--
-- Non-zero exit code on any mismatch.
module Main where

import Circuit.Channel (trace)
import Circuit.Learn.Adam
  ( adam,
    adamDecomposed,
    adamReference,
    adamUpdates,
    ewma,
    ewmaDirect,
  )
import Circuit.Learn.Ephemeral
  ( Experience (..),
    Progress (..),
    Task (..),
    learn,
    sgd,
  )
import Circuit.Learn.Fit (fit, forward, loss, toyData)
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
-- Two-EWMA Adam decomposition
-- ---------------------------------------------------------------------------

checkAdamDecomposed :: IO Bool
checkAdamDecomposed = do
  let alpha = 0.01
      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8

  -- 1. Decomposed Adam vs direct Adam on same trace
  let direct = scan (adam alpha beta1 beta2 eps) gradTrace
      decomposed = adamDecomposed alpha beta1 beta2 eps gradTrace
      ok1 = allNear epsLoose direct decomposed
  report "Adam decomposed == direct (output)" ok1

  -- 2. m channel of reference matches independent EWMA of gradients
  --    Adam uses beta1 as OLD-value weight; ewmaDirect uses alpha as NEW-value weight.
  --    So ewmaDirect (1-beta1) 0 == Adam's m channel.
  let (g0, gs) = case gradTrace of (x : xs) -> (x, xs); [] -> error "empty trace"
      refStates = adamReference beta1 beta2 g0 gs
      refM = map (\(m, _, _) -> m) refStates
      directM = ewmaDirect (1 - beta1) 0 gradTrace
      ok2 = allNear epsTight refM directM
  report "Adam m channel == EWMA of gradients" ok2

  -- 3. v channel of reference matches independent EWMA of squared gradients
  let refV = map (\(_, v, _) -> v) refStates
      directV = ewmaDirect (1 - beta2) 0 (map (** 2) gradTrace)
      ok3 = allNear epsTight refV directV
  report "Adam v channel == EWMA of squared gradients" ok3

  pure (ok1 && ok2 && ok3)

-- ---------------------------------------------------------------------------
-- Toy 2-layer NN fit
-- ---------------------------------------------------------------------------

checkToyFit :: IO Bool
checkToyFit = do
  let -- Initial weights: small random-like values
      w0 = [0.1, -0.2, 0.05, -0.05, 0.3, -0.1, 0.0] :: [Double]
      steps = 200

  -- Run fit
  let (_wN, initLoss, finalLoss) = fit steps w0 toyData

  -- 1. Loss decreases
  let ok1 = finalLoss < initLoss
  report "Toy fit: loss decreases" ok1

  -- 2. Forward pass produces finite output
  let y0 = forward w0 1.0
      ok2 = not (isNaN y0 || isInfinite y0)
  report "Toy fit: forward pass finite" ok2

  -- 3. Loss is non-negative
  let ok3 = initLoss >= 0 && finalLoss >= 0
  report "Toy fit: loss non-negative" ok3

  pure (ok1 && ok2 && ok3)

-- ---------------------------------------------------------------------------
-- Ephemeral Progress / Experience framing
-- ---------------------------------------------------------------------------

-- | Linear model @y = m*x + b@ with parameters @[m, b]@.
linearForward :: [Double] -> Double -> Double
linearForward [m, b] x = m * x + b
linearForward _ _ = error "linearForward: expected [m, b]"

-- | Squared-error loss for one example.
linearLoss :: [Double] -> (Double, Double) -> Double
linearLoss params (x, y) = (linearForward params x - y) ** 2

-- | Gradient of squared error for one example.
linearGrad :: [Double] -> (Double, Double) -> [Double]
linearGrad [m, b] (x, y) =
  let yPred = m * x + b
      err = yPred - y
   in [2 * err * x, 2 * err]
linearGrad _ _ = error "linearGrad: expected [m, b]"

-- | Total loss over a dataset.
totalLoss :: [Double] -> [(Double, Double)] -> Double
totalLoss params dataset = sum (map (linearLoss params) dataset)

-- | Check that folding SGD progress over an experience set decreases loss.
checkEphemeralSGD :: IO Bool
checkEphemeralSGD = do
  let dataset = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
      experience = Experience dataset
      prog = sgd 0.05 linearGrad
      params0 = [0.0, 0.0]
      paramsN = iterate (learn prog experience) params0 !! 100
      initLoss = totalLoss params0 dataset
      finalLoss = totalLoss paramsN dataset
      ok = finalLoss < initLoss && finalLoss < 0.01
  report ("Ephemeral SGD loss (init=" ++ show initLoss ++ ", final=" ++ show finalLoss ++ ")") ok
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
  okAdamDec <- checkAdamDecomposed
  okToyFit <- checkToyFit
  okEphemeralSGD <- checkEphemeralSGD
  okAssoc <- checkAssoc
  okId <- checkIdentity
  okConstState <- checkConstantState
  let allOk = and [okEWMA, okAdam, okAdamDec, okToyFit, okEphemeralSGD, okAssoc, okId, okConstState]
  if allOk
    then putStrLn "circuits-learn axioma: all checks passed"
    else do
      putStrLn "circuits-learn axioma: one or more checks failed"
      exitFailure
