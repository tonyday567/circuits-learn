{-# LANGUAGE DerivingStrategies #-}

-- | Adam as a 'Process' and as a reference recurrence.
--
-- The frontier claim is that Adam decomposes into two EWMAs plus a pointwise
-- quotient.  This module starts with a direct Moore-machine Adam (with bias
-- correction) and verifies its arithmetic against a hand recurrence on a fixed
-- gradient trace.
module Circuit.Learn.Adam
  ( -- * EWMA building block
    ewma,

    -- * Adam
    adam,
    adamReference,
    adamUpdates,
  )
where

import Circuit.Process (Process (..), scan)
import Data.List (scanl')

-- | Exponentially weighted moving average as a 'Process'.
--
-- State is the current EWMA value; output is the same value.
ewma :: Double -> Double -> Process Double Double
ewma alpha s0 = Process inject step extract
  where
    inject x0 = alpha * x0 + (1 - alpha) * s0
    step s x = alpha * x + (1 - alpha) * s
    extract s = s

-- | Adam parameter update as a 'Process'.
--
-- Input: gradient. Output: parameter update (not the new parameter).
-- State: @(m, v, t)@ — first moment, second moment, timestep.
adam :: Double -> Double -> Double -> Double -> Process Double Double
adam alpha beta1 beta2 eps = Process inject step extract
  where
    inject g0 =
      let m1 = (1 - beta1) * g0
          v1 = (1 - beta2) * g0 * g0
       in (m1, v1, 1)
    step (m, v, t) g =
      let t' = t + 1
          m' = beta1 * m + (1 - beta1) * g
          v' = beta2 * v + (1 - beta2) * g * g
       in (m', v', t')
    extract (m, v, t) =
      let mHat = m / (1 - beta1 ** t)
          vHat = v / (1 - beta2 ** t)
       in alpha * mHat / (sqrt vHat + eps)

-- | Reference Adam state recurrence on a gradient trace.
--
-- Forward recurrence starting from @(0, 0, 0)@; the first element corresponds
-- to the state after observing @g0@.
adamReference :: Double -> Double -> Double -> [Double] -> [(Double, Double, Int)]
adamReference beta1 beta2 g0 gs = drop 1 $ scanl' step (0, 0, 0) (g0 : gs)
  where
    step (m, v, t) g =
      let t' = t + 1
          m' = beta1 * m + (1 - beta1) * g
          v' = beta2 * v + (1 - beta2) * g * g
       in (m', v', t')

-- | Convert reference states to Adam parameter updates.
--
-- This applies the same bias-correction and scaling as 'adam' so the two can
-- be compared directly on a fixed trace.
adamUpdates :: Double -> Double -> Double -> Double -> [(Double, Double, Int)] -> [Double]
adamUpdates alpha beta1 beta2 eps = map update
  where
    update (m, v, t) =
      let mHat = m / (1 - beta1 ** fromIntegral t)
          vHat = v / (1 - beta2 ** fromIntegral t)
       in alpha * mHat / (sqrt vHat + eps)
