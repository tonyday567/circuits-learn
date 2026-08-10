-- | Parameterised morphisms: @(p, a) -> b@.
--
-- @Para p@ is the constant-state slice of @Loop (,)@ — it threads a read-only
-- parameter through a composition chain.  Composition threads the same
-- parameter to both morphisms; 'id' discards it.
--
-- === Law oracles (see circuits-learn-axioma)
--
-- 1. 'Category' associativity: @(h '.' g) '.' f == h '.' (g '.' f)@
-- 2. @Para@ composition is @Loop (,)@ constant-state: threading a constant
--    parameter through a loop yields the same result as the para composition.
module Circuit.Learn.Para
  ( -- * Type
    Para (..),

    -- * Running
    runPara,
    liftPara,
    forgetPara,
  )
where

import Control.Arrow (Arrow (..), ArrowLoop (..))
import Control.Category (Category (..))
import Data.Profunctor (Costrong (..), Profunctor (..), Strong (..))
import Prelude hiding (id, (.))

-- | Parameterised morphism: @(p, a) -> b@.
newtype Para p a b = Para {unPara :: (p, a) -> b}

-- | Run with explicit parameter.
runPara :: Para p a b -> p -> a -> b
runPara (Para f) p a = f (p, a)

instance Category (Para p) where
  id = Para snd
  Para g . Para f = Para $ \(p, a) -> g (p, f (p, a))

instance Arrow (Para p) where
  arr f = Para $ \(_, a) -> f a
  first (Para f) = Para $ \(p, (a, c)) -> (f (p, a), c)

instance ArrowLoop (Para p) where
  loop (Para f) = Para $ \(p, a) ->
    let (b, c) = f (p, (a, c)) in b

instance Profunctor (Para p) where
  dimap f g (Para m) = Para $ \(p, a) -> g (m (p, f a))

instance Strong (Para p) where
  first' (Para f) = Para $ \(p, (a, c)) -> (f (p, a), c)

instance Costrong (Para p) where
  unfirst (Para f) = Para $ \(p, a) ->
    let (b, c) = f (p, (a, c)) in b

-- | Lift a plain function, ignoring the parameter.
liftPara :: (a -> b) -> Para p a b
liftPara f = Para $ \(_, a) -> f a

-- | Forget the parameter, recover plain function.
forgetPara :: p -> Para p a b -> a -> b
forgetPara p (Para f) a = f (p, a)
