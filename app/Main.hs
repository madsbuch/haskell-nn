module Main where

import MnistLoader
import Network

import Numeric.LinearAlgebra as L

main :: IO ()
main = testMnist

testMnist = do
  net <- initNet [784, 30, 10]
  exemplars <- loadDataWrapper
  test <- loadTestWrapper

  -- let eta = 0.050
  -- let batchSize = 20
  -- let epochs = 30

  let eta = 0.5
  let batchSize = 16
  let epochs = 30

  putStrLn "Running Stochastic Gradient Descent "
  putStrLn $ "eta: " ++ (show eta)
  putStrLn $ "Batch Size: " ++ (show batchSize)
  putStrLn $ "Epochs: " ++ (show epochs)

  trainedNetwork <- sgd net exemplars test epochs 0 batchSize eta
  return ()

test = do
  net <- initNet [3, 5]
  
  putStrLn "The Net"
  putStrLn $ show net
  
  putStrLn "Feed Forward"
  putStrLn $ show $ feedForward net (vect 3 [0, 1, 0])
  
  putStrLn "A ForwardPass"
  putStrLn $ show $ forwardPass net (vect 3 [0, 1, 0])
  
  putStrLn "Entire backprop"
  let eb = backprop net (vect 3 [0, 1, 0]) (vect 5 [1, 0, 1, 0, 1])
  putStrLn $ show $ length eb
  putStrLn $ show $ eb

  putStrLn "A Minibatch"
  let x = vect 3 [0,1,0]
  let y = vect 5 [1, 0, 1, 0, 1]
  let batch = [(x, y), (x, y), (x, y)]
  let newn = updateMiniBatch net batch 3.0
  putStrLn $ show newn
  return newn

vect n = (n><1)