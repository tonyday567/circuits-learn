-- | Toy 2-layer neural network fit.
--
-- A small regression example: 1→2→1 network with tanh activation, trained
-- by Adam-style gradient descent to approximate @y = 2*x@.
--
-- The oracle in circuits-learn-axioma asserts that loss decreases
-- over training: @loss(weights_0) > loss(weights_N)@.
module Circuit.Learn.Fit
  ( -- * Network
    forward,
    loss,

    -- * Training
    fit,

    -- * Dev dataset
    toyData,
  )
where

import Prelude

-- ---------------------------------------------------------------------------
-- 2-layer network: 1 → 2 → 1 with tanh activation
-- 7 params: w1,w2,b1,b2,w3,w4,b3
-- ---------------------------------------------------------------------------

-- | tanh activation.
tanhAct :: Double -> Double
tanhAct x = (exp x - exp (-x)) / (exp x + exp (-x))

-- | Forward pass: 1 → 2 → 1.
forward :: [Double] -> Double -> Double
forward [w1, w2, b1, b2, w3, w4, b3] x =
  let h1 = tanhAct (w1 * x + b1)
      h2 = tanhAct (w2 * x + b2)
   in w3 * h1 + w4 * h2 + b3
forward _ _ = error "forward: expected exactly 7 params"

-- | Mean squared error loss.
loss :: [Double] -> [(Double, Double)] -> Double
loss w dataset =
  let total = sum [(forward w x - y) ** 2 | (x, y) <- dataset]
   in total / fromIntegral (length dataset)

-- | Toy dataset: y ≈ 2*x with slight noise on the last point.
toyData :: [(Double, Double)]
toyData = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.5)]

-- ---------------------------------------------------------------------------
-- Training loop — Adam-style gradient descent
-- ---------------------------------------------------------------------------

-- | Compute the gradient via central finite differences.
grad :: [Double] -> [(Double, Double)] -> [Double]
grad w dataset =
  let base = loss w dataset
      h = 1e-5
      perturb i sign =
        let w' = zipWith (+) w (replicate i 0 ++ [sign * h] ++ repeat 0)
         in sign * (loss w' dataset - base) / h
   in [perturb i 1 | i <- [0 .. 6]]

-- | Adam step on a (params, m, v, t) state tuple.
adamStep ::
  Double ->
  Double ->
  Double ->
  [(Double, Double)] ->
  ([Double], [Double], [Double], Int) ->
  ([Double], [Double], [Double], Int)
adamStep beta1 beta2 alpha dataset (w, m, v, t) =
  let g = grad w dataset
      t' = t + 1
      m' = zipWith (\mi gi -> beta1 * mi + (1 - beta1) * gi) m g
      v' = zipWith (\vi gi -> beta2 * vi + (1 - beta2) * gi * gi) v g
      mHat = map (/ (1 - beta1 ** fromIntegral t')) m'
      vHat = map (/ (1 - beta2 ** fromIntegral t')) v'
      update = zipWith (\mh vh -> alpha * mh / (sqrt vh + eps)) mHat vHat
      eps = 1e-8
   in (zipWith (-) w update, m', v', t')

-- | Run N steps of Adam, returning trained params and initial/final loss.
fit ::
  Int ->
  [Double] ->
  [(Double, Double)] ->
  ([Double], Double, Double)
fit steps w0 dataset =
  let beta1 = 0.9
      beta2 = 0.999
      alpha = 0.05
      initLoss = loss w0 dataset
      initState = (w0, replicate 7 0, replicate 7 0, 0 :: Int)
      (wN, _, _, _) = iterate (adamStep beta1 beta2 alpha dataset) initState !! steps
      finalLoss = loss wN dataset
   in (wN, initLoss, finalLoss)
