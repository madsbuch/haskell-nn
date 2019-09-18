{-# LANGUAGE FlexibleContexts #-}
module Network (initNet, feedForward, sgd, forwardPass, backprop, updateMiniBatch) where

import Control.Monad
import Data.List.Split
import Numeric.LinearAlgebra as L

import MnistLoader
import Aux

feedForward :: Network -> Matrix Float -> Matrix Float
feedForward n a = foldl f a $ zip (weights n) (biases n)
  where
    f a' (w, b) = sigmoid ((w L.<> a') + b)

sgd :: Network
  -> [(Matrix Float, Matrix Float)] -- traning data
  -> [(Matrix Float, Matrix Float)] -- Evaluation data
  -> Int -- Number of epochs
  -> Int -- Current Epoc
  -> Int -- Mini batch size
  -> Float -- Eta
  -> IO Network
sgd n exemplars testData maxEpoch epoch batchSize eta = do
  exemplars' <- shuffle exemplars

  let batches = chunksOf batchSize exemplars'
  let newNet = foldl (\accNet batch -> updateMiniBatch accNet batch eta) n batches
  let correct = evaluate newNet testData
  putStrLn $ "Epoch " ++ (show (epoch + 1)) ++ " Eval: " ++ (show correct) ++ " / " ++ (show $ length testData)

  if epoch == maxEpoch
    then return newNet
    else sgd newNet exemplars' testData maxEpoch (epoch + 1) batchSize eta

updateMiniBatch :: Network -> [(Matrix Float, Matrix Float)] -> Float -> Network
updateMiniBatch n l@((x, y) : rest) eta = let 
    initAcc = backprop n x y 
    nablaSums = foldl f initAcc rest
    nabla_b = map fst nablaSums
    nabla_w = map snd nablaSums
  in
    Network {
        sizes = sizes n
      , weights = zipWith wavg (weights n) nabla_w
      , biases = zipWith wavg (biases n) nabla_b
    }
  where
    f nablaSum (x, y) = let
        bp = backprop n x y
      in
        zipWith add nablaSum bp
    add (nb, nw) (nb', nw') = (nb+nb', nw+nw')
    wavg :: Matrix Float -> Matrix Float -> Matrix Float
    wavg m1 m2 = let 
        nt = (1><1) [fromIntegral $ length l]
        eta' = (1><1) [eta]
      in m1 - (eta' / nt) * m2


-- Network -> x -> y -> (nabla_b, nabla_w)
backprop :: Network -> Matrix Float -> Matrix Float -> [(Matrix Float, Matrix Float)]
backprop n x y = let
    -- Reversed list of forward passes
    ((a:a':as), (z:zs)) = forwardPass n x
    revW = foldl (\acc w -> w:acc) [] (weights n)
    fp = zip as zs
    zippedFp = zip fp revW
    
    -- Handle last layer manually
    delta = (cost_derivative a y) * (sigmoid' z)
    nabla_b = delta
    nabla_w = delta L.<> (tr' a')
    res = foldl f [(nabla_b, nabla_w)] zippedFp
  in
    res
  where
    f (acc@((delta, _):_)) ((a, z), w) = let
        sp = sigmoid' z
        delta' = ((tr' w) L.<> delta) * sp
        nabla_b = delta'
        nabla_w = delta' L.<> (tr' a)
      in
        (nabla_b, nabla_w) : acc

-- Network -> activation -> (activations, z vectors)
forwardPass :: Network -> Matrix Float -> ([Matrix Float], [Matrix Float])
forwardPass n a = let
  pairs = zip (weights n) (biases n)
  fp = foldl f [(a, a)] pairs
  in
    (map fst fp, init $ map snd fp)
    where
      f (l@((a, _):_)) (w, b) = let
          z = (w L.<> a) + b
          activation = sigmoid z
        in (activation, z):l

evaluate :: Network -> [(Matrix Float, Matrix Float)] -> Int
evaluate net exemplars = sum $ map f exemplars
  where
    f (x, y) = let
        approx = argMax $ feedForward net x
        actual = argMax y 
      in
        if approx == actual
          then 1
          else 0

cost_derivative output_activations y = output_activations - y

-- Miscellaneous
sigmoid z = 1.0 / (1.0 + (exp $ -z))

sigmoid' :: Matrix Float -> Matrix Float
sigmoid' z = (sigmoid z) * (1 - (sigmoid z))

argMax :: Matrix Float -> Int
argMax m = L.maxIndex $ L.flatten m

-- assuming row-major order
data Network = Network { sizes :: [Int]
                       , weights :: [Matrix Float]
                       , biases :: [Matrix Float]
                       } deriving (Show)

initNet :: [Int] -> IO (Network)
initNet sizes = do
  ws <- genWeights sizes
  bs <- genBiases sizes
  return Network { sizes = sizes
                  , weights = ws
                  , biases = bs
                  }
  where
    genWeights [s, s'] =
      do
        m <- L.randn s' s
        return [L.cmap realToFrac m]
    genWeights (s:s':ss) = 
      do
        rest <- genWeights (s' : ss)
        m <- L.randn s' s
        return (L.cmap realToFrac m : rest)
    genWeights _ = error "Not enough layers"
    genBiases [s, s'] =
      do
        m <- L.randn s' 1
        return $ [L.cmap realToFrac m]
    genBiases (s:s':ss) = 
      do
        rest <- genBiases (s' : ss)
        m <- L.randn s' 1
        return (L.cmap realToFrac m : rest)
    genBiases _ = error "Not enough layers"