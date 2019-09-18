module Aux (shuffle, samples) where

import qualified System.Random as SR
import qualified Data.Random as DR
import Data.Array.IO
import Control.Monad

shuffle :: [a] -> IO [a]
shuffle xs = do
        ar <- newArray n xs
        forM [1..n] $ \i -> do
            j <- SR.randomRIO (i,n)
            vi <- readArray ar i
            vj <- readArray ar j
            writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (IOArray Int a)
    newArray n xs =  newListArray (1,n) xs

samples n = DR.runRVar (replicateM n (DR.stdNormal)) DR.StdRandom :: IO [Double]
