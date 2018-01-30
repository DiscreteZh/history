package com.enn

import com.enn.App.log
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, MultilabelMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.collection.mutable.ArrayBuffer

class Evaluation {

  /**
    * 计算分类评估指标
    * @param predictAndLabel :预测结果和标签值组成的RDD
    * @return ：
    */
  def getWantedEVA(predictAndLabel:RDD[(Double,Double)], sqlContext:SQLContext)={
    log.info("正在进行模型评估指标计算！！！")
    val metrics = new MulticlassMetrics(predictAndLabel)
    val accuracy = getAccuracy(predictAndLabel)
    val confusionMatrix = metrics.confusionMatrix
    println(confusionMatrix)
    val rows = confusionMatrix.numRows
    val columns = confusionMatrix.numCols
    val MulityConf = new ArrayBuffer[Array[Double]]
    val arrPrecision = new ArrayBuffer[Double]
    val arrCall = new ArrayBuffer[Double]
    val arrLabels = new ArrayBuffer[Double]
    for (i <- 0 until rows){
      val precision = metrics.precision(i)
      val recall = metrics.recall(i)
      val tempConfM = new ArrayBuffer[Double]
      for (j<-0 until columns){
        tempConfM += confusionMatrix(i,j)
      }
      MulityConf += tempConfM.toArray
      arrPrecision += precision
      arrCall += recall
      arrLabels += i
    }
    (arrLabels.toArray, MulityConf.toArray, arrPrecision.toArray, arrCall.toArray, accuracy)
  }

  /**
    * 计算多分类的准确率
    * @param input:输入RDD(predictions, labels, )
    * @return
    */
  def getAccuracy(input:RDD[(Double,Double)])={
    val predictAndLabel = input.map(x=>{
      (Array(x._1),Array(x._2))
    })
    val metrics = new MultilabelMetrics(predictAndLabel)
    metrics.accuracy
  }

  /**
    *
    * @param predictAndLabel:
    * @param N:
    * @return
    */

  def getBinary(predictAndLabel:RDD[(Double, Double)], N:Long)={
    log.info("============================开始计算评估指标！！！")
    val accuracy = (1.0 * predictAndLabel.filter(x => x._1 == x._2).count()/ N)
    val metrics = new MulticlassMetrics(predictAndLabel)
    val confusionMatrix = metrics.confusionMatrix
    val rows = confusionMatrix.numRows
    val columns = confusionMatrix.numCols
    val MulityConf = new ArrayBuffer[Array[Double]]
    val arrPrecision = new ArrayBuffer[Double]
    val arrCall = new ArrayBuffer[Double]
    val arrLabels = new ArrayBuffer[Double]
    for (i <- 0 until rows){
      val precision = metrics.precision(i)
      val recall = metrics.recall(i)
      val tempConfM = new ArrayBuffer[Double]
      for (j<-0 until columns){
        tempConfM += confusionMatrix(i,j)
      }
      MulityConf += tempConfM.toArray
      arrPrecision += precision
      arrCall += recall
      arrLabels += i
    }
    (arrLabels.toArray, MulityConf.toArray, arrPrecision.toArray, arrCall.toArray, accuracy)
  }
}
