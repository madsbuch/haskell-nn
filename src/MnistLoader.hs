module MnistLoader (loadData, loadDataWrapper, loadTestWrapper, investigate) where

import Data.IDX
import qualified Data.Vector.Unboxed as V
import Numeric.LinearAlgebra as L

vect n = (n><1)

loadData :: IO ([(Int, V.Vector Double)])
loadData = do
  mData <- decodeIDXFile "data/train-images-idx3-ubyte"
  let d = getSomething mData
  mLabels <- decodeIDXLabelsFile "data/train-labels-idx1-ubyte"
  let l = getSomething mLabels
  return $ getSomething $ labeledDoubleData l d

loadTest :: IO ([(Int, V.Vector Double)])
loadTest = do
  mData <- decodeIDXFile "data/t10k-images.idx3-ubyte"
  let d = getSomething mData
  mLabels <- decodeIDXLabelsFile "data/t10k-labels.idx1-ubyte"
  let l = getSomething mLabels
  return $ getSomething $ labeledDoubleData l d

-- Converts the label component to a vector representation where
-- n'th component is 1
loadDataWrapper :: IO ([(Matrix Float, Matrix Float)])
loadDataWrapper = do
  theData <- loadData
  return $ map (\(l, d) -> (mVect d, makeVector l)) theData
  where
    makeVector l = vect 10 $ (map (\i -> if i == l then 1 else 0) [0..10])
    mVect l = vect (V.length l) (map realToFrac $ V.toList l)

loadTestWrapper :: IO ([(Matrix Float, Matrix Float)])
loadTestWrapper = do
  theData <- loadTest
  return $ map (\(l, d) -> (mVect d, makeVector l)) theData
  where
    makeVector l = vect 10 $ (map (\i -> if i == l then 1 else 0) [0..10])
    mVect l = vect (V.length l) $ map realToFrac $ V.toList l

investigate :: IO ()
investigate = do
    putStrLn "Inspecting File"
    training_data_mb <- decodeIDXFile "data/train-images-idx3-ubyte"
    let td = getSomething training_data_mb

    putStrLn "Image Data"
    putStr "idxType: "
    putStrLn $ show $ idxType td

    putStr "idxDimensions: "
    putStrLn $ show $ idxDimensions td

    putStr "isIDXReal: "
    putStrLn $ show $ isIDXReal td

    putStr "isIDXIntegral: "
    putStrLn $ show $ isIDXIntegral td

    let training_vector = idxDoubleContent td
    putStrLn $ show $ V.length training_vector

    loaded_data <- loadData
    let (l, d) = loaded_data !! 1
    putStrLn $ show $ V.length $ d
    putStrLn $ show $ l

getSomething :: Maybe a -> a
getSomething (Just x) = x
getSomething Nothing = error "Bah"