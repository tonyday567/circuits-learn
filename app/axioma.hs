-- | Axioma oracle for the circuits-learn supervised slice.
--
-- Verifies that 'Process'-based optimisers agree with hand/reference
-- recurrences on a fixed trace. Non-zero exit code on mismatch.
module Main where

import Circuit.Learn.Adam (adam, adamReference, adamUpdates, ewma)
import Circuit.Process (scan)
import Data.List (scanl')
import System.Exit (exitFailure)

-- | A short, deterministic gradient trace.
trace :: [Double]
trace = [0.5, -0.3, 0.2, -0.8, 0.1, 0.4, -0.2]

-- | Exact EWMA recurrence for comparison.
ewmaReference :: Double -> Double -> [Double] -> [Double]
ewmaReference _ _ [] = []
ewmaReference alpha s0 (x0 : xs) = scanl' step (alpha * x0 + (1 - alpha) * s0) xs
  where
    step s x = alpha * x + (1 - alpha) * s

-- | Tolerance for floating-point comparisons near zero.
epsTight :: Double
epsTight = 1e-12

-- | Tolerance for bias-corrected Adam values, where rounding error accumulates.
epsLoose :: Double
epsLoose = 1e-10

near :: Double -> Double -> Double -> Bool
near tol a b = abs (a - b) <= tol

allNear :: Double -> [Double] -> [Double] -> Bool
allNear tol as bs = length as == length bs && and (zipWith (near tol) as bs)

report :: String -> Bool -> IO ()
report label ok = putStrLn $ label ++ ": " ++ if ok then "PASS" else "FAIL"

checkEWMA :: IO Bool
checkEWMA = do
  let alpha = 0.9
      s0 = 0.0
      expected = ewmaReference alpha s0 trace
      actual = scan (ewma alpha s0) trace
      ok = allNear epsTight expected actual
  report "EWMA vs hand recurrence" ok
  pure ok

checkAdam :: IO Bool
checkAdam = do
  let alpha = 0.01
      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8
      (g0, gs) = case trace of (x : xs) -> (x, xs); [] -> error "empty trace"
      expected = adamUpdates alpha beta1 beta2 eps (adamReference beta1 beta2 g0 gs)
      actual = scan (adam alpha beta1 beta2 eps) trace
      ok = allNear epsLoose expected actual
  report "Adam vs reference recurrence" ok
  pure ok

main :: IO ()
main = do
  okEWMA <- checkEWMA
  okAdam <- checkAdam
  if okEWMA && okAdam
    then putStrLn "circuits-learn axioma: all checks passed"
    else do
      putStrLn "circuits-learn axioma: one or more checks failed"
      exitFailure
