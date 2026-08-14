-- | Ephemeral learning vocabulary adapted to the circuits-learn interface.
--
-- Based on ideas from the @ephemeral@ package: a learning step is a
-- 'Progress' that transduces a parameterised 'Task' given an 'Experience'.
-- Folding progress over an experience set is 'learn'; choosing the better
-- resulting task is 'improve'.
module Circuit.Learn.Ephemeral
  ( -- * Core vocabulary
    Task (..),
    Experience (..),
    Progress (..),
    Learn (..),

    -- * Learning as folding progress
    learn,
    improve,

    -- * Simple gradient descent progress
    sgd,
  )
where

import Data.Foldable (Foldable (..))

-- | A task is a parameterised measurement: given parameters @p@ and an
-- experience @e@, produce a performance value @r@.
--
-- This is the circuits-learn reading of ephemeral's @Task e p@, with the
-- parameter block @p@ made explicit so it can be updated by 'Progress'.
newtype Task p e r = Task
  { measure :: p -> e -> r
  }

-- | An experience is a container of training examples.
newtype Experience f e = Experience
  { set :: f e
  }

-- | A progress step updates parameters from one experience.
--
-- In the original ephemeral vocabulary this transduces the task itself; here
-- the task is fixed and the parameters that define it are changed.
newtype Progress p e = Progress
  { step :: e -> p -> p
  }

-- | A learn folds a 'Progress' over an 'Experience' set.
newtype Learn f e p = Learn
  { change :: Experience f e -> p -> p
  }

-- | Fold a progress step over all experiences in a set.
learn :: (Foldable f) => Progress p e -> Experience f e -> p -> p
learn p (Experience es) params0 = foldl' (\params e -> step p e params) params0 es

-- | Apply a learn to a task and choose the better parameters.
--
-- Performance is summarised by a user-supplied function @f (r) -> r@ (for
-- example 'sum' or 'mean') so that two parameter blocks can be compared.
improve ::
  (Foldable f, Functor f, Ord r) =>
  (f r -> r) ->
  Progress p e ->
  Experience f e ->
  Task p e r ->
  p ->
  p
improve summarize prog es (Task measure) params0 =
  let params1 = learn prog es params0
      perf0 = summarize (measure params0 <$> set es)
      perf1 = summarize (measure params1 <$> set es)
   in if perf1 > perf0 then params1 else params0

-- | Plain stochastic-gradient-descent progress for vector parameters.
--
-- @sgd rate grad@ takes one example, computes the gradient @grad params e@,
-- and steps the parameters in the negative direction.
sgd ::
  -- | learning rate
  Double ->
  -- | gradient of the task at parameters and example
  ([Double] -> e -> [Double]) ->
  Progress [Double] e
sgd rate grad = Progress $ \e params ->
  let g = grad params e
   in zipWith (-) params (map (rate *) g)
