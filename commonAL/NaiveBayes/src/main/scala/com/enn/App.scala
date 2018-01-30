package com.enn

import java.io.FileNotFoundException

import com.alibaba.fastjson.JSONObject
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by LYY on 2017/1/3.
  */
object App {
  val log = Logger.getLogger("AppWithNaiveBayesClassfication")
  /**
    * @param args
    * action：  包含：train/test/predictions
    * taskId：  任务Id
    * resultPath： 结果保存路径
    * modeFileName：模型文件路径
    * soucePathFile：源文件路径
    * destPath: 输出数据保存路径
    * selectCols：选择的列集合（是一个数组）
    * allCols: 所有字段名称集合
    * label：label标签
    * split：原始数据集分割方式x
    * coreMax: 最大核数
    */
  def main(args:Array[String]):Unit={
    val TRAIN="train"
    val TEST="test"
    val PREDICTION = "predictions"
    val params = new mutable.HashMap[String,String]()
    val inputArgs=Array(
      "action=train",
      "taskId=train15154026537330b9207ff039048fb91166376920b86b0",
      "hdfsRootPath=hdfs://node86:8020/",
      "resultPathFile=/user/spark/result//train15154026537330b9207ff039048fb91166376920b86b0",
      "modelPathFile=/user/spark/model//model1515402653733bb01972bda174c6ab83525bd1132f67e",
      "dataSetPathFile=/user/hive/warehouse/train_1515401194555",
      "allCols=ddatem_f183d3a2c4ea48f2ac4b4948d60d884d,label_d413de5ae4ac4890b5d05e7f102ddbd7,ddagfhtew_dfbf8bbbd683413c87bc8cf9cbb1038d,unitpgfhriceusd_f324f0f35f9e4d8bbc733b4a49aca5f4,gcertflag_c3339ef8c6a84caf93225ddd56c78c9d,ddagfhtgfheh_e9500bf889ba4cf2b8183e9ec42b4f1e,ddatemgfhgfinute_d306b76ff8e4430c8a43440a98037a6d,dgfhdated_c28b3a937ae84383ae14dfffbcfb652a",
      "selectCols=gcertflag_c3339ef8c6a84caf93225ddd56c78c9d,ddatem_f183d3a2c4ea48f2ac4b4948d60d884d,dgfhdated_c28b3a937ae84383ae14dfffbcfb652a,ddagfhtew_dfbf8bbbd683413c87bc8cf9cbb1038d,ddagfhtgfheh_e9500bf889ba4cf2b8183e9ec42b4f1e",
      "label=label_d413de5ae4ac4890b5d05e7f102ddbd7",
      "split=\u0001",
      "runCoreMax=20")
    inputArgs.foreach(x=>{
      val arr = x.split("=")
      if (arr.size > 1) {
        params.put(arr(0), arr(1))
      }
    })
    val hdfsRootPath = params.get("hdfsRootPath").get
    val action=if(!params.get("action").isEmpty) params.get("action").get else throw new IllegalArgumentException("action (train,test or predictions) must be choose!")
    val taskId = params.get("taskId").get
    val resultPath=if (!params.get("resultPathFile").isEmpty) (hdfsRootPath + params.get("resultPathFile").get.trim) else throw new FileNotFoundException("resultPath must be set!")
    val modeFileName=if (!params.get("modeFileName").isEmpty) (hdfsRootPath + params.get("modeFileName").get.trim) else throw new FileNotFoundException("modeFileName must be set!")
    val dataSetPathFile =if (!params.get("dataSetPathFile").isEmpty) (hdfsRootPath +params.get("dataSetPathFile").get.trim) else throw new FileNotFoundException("dataSetPathFile must be set!")
    val destPath =if (!params.get("destPath").isEmpty) (hdfsRootPath + params.get("destPath").get) else ""
    val labelCol = if(!params.get("labelColName").isEmpty) params.get("labelColName").get else null
    val selectCols = params.get("selectCols").get.split(",")
    var allCols = Array[String]()
    if(!params.get("allCols").isEmpty){
      allCols = params.get("allCols").get.split(",")
    }
    val splitType = if(!params.get("split").isEmpty) params.get("split").get else ","
    val coreMax = params.get("runCoreMax").get
      Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
      Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
      log.info("正在执行NaiveBayes算法!")
      val conf = new SparkConf()
        .set("spark.cores.max",coreMax)
        .setAppName(taskId)
        .setMaster("local")
    conf.set("spark.testing.memory", "2147480000")
      val sc = new SparkContext(conf)
      try{
        if(TRAIN.equals(action)){
          train(sc, dataSetPathFile,allCols,labelCol, selectCols,splitType,modeFileName, resultPath)
        }else if(TEST.equals(action)){
          test(sc,dataSetPathFile, allCols, selectCols, labelCol,splitType,modeFileName, destPath, resultPath)
        }else if(PREDICTION.equals(action)){
          predict(sc,dataSetPathFile, allCols ,selectCols,splitType ,modeFileName,destPath, resultPath)
        }
      }finally {
        sc.stop()
      }
  }
  /**
    *
    * @param sc: SparkContext
    * @param filePath: 加载数据集路径
    * @param headers:原始数据表头
    * @param labelCol:标签列名
    * @param selectCols：选取用于训练的特征
    * @param delimiter：原始数据分隔方式
    * @param modePath：模型保存路径
    * @param resultPath：结果集保存路径
    * @return
    */
  def train(sc:SparkContext,
            filePath:String,
            headers:Array[String],
            labelCol: String,
            selectCols:Array[String],
            delimiter:String,
            modePath:String,
            resultPath:String)={
    case class Result(status: String,causeBy:String,applicationID:String){
      def make()={
        val json = new JSONObject()
        json.put("status", status)
        json.put("msg", causeBy)
        json.put("applicationID", applicationID)
        json
      }
    }
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val du = new DataUtils()
    val wr = new WriteResults()
    val numIterations = 100
    var result = new JSONObject()
    try{
      require(!(labelCol==null),"label must be set!")
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers, delimiter)
      orgDataDF.show(10, false)
      val labeledData = du.toLabeledPoints(sqlContext,orgDataDF, selectCols,labelCol)
      log.info("================================开始模型训练！！！")
      val model = NaiveBayes.train(labeledData)
      if ( model != null){
        log.info("================================开始保存训练结果！！！")
        model.save(sc,modePath)
        result = Result("0", null, sc.applicationId).make()
      }
    }catch{
      case ex: Exception =>{
        ex.printStackTrace()
        result = Result("1",ex.getMessage,null).make()
      }
    }
    wr.writeResult(resultPath, result.toJSONString)
  }
  /**
    *
    * @param sc: SparkContext
    * @param filePath: 加载数据集路径
    * @param headers:原始数据表头
    * @param labelCol:标签列名
    * @param selectCols：选取用于训练的特征
    * @param delimiter：原始数据分隔方式
    * @param modePath：模型保存路径
    * @param resultPath：结果集保存路径
    * @return
    */
  def test(sc:SparkContext,
           filePath:String,
           headers:Array[String],
           selectCols:Array[String],
           labelCol: String,
           delimiter:String,
           modePath:String,
           destPath:String,
           resultPath:String)={
    case class Reslut(status: String,
                      causeBy:String,
                      evaluation: JSONObject,
                      applicationID:String,
                      headers:Array[String],
                      scoreColName:String,
                      predictionColName:String,
                      result: String){
      def make()={
        val json = new JSONObject()
        json.put("status", status)
        json.put("msg", causeBy)
        json.put("applicationID", applicationID)
        val obj = new JSONObject()
        obj.put("evaluation", evaluation)
        obj.put("headers", headers)
        obj.put("scoreColName", scoreColName)
        obj.put("predictionColName", predictionColName)
        obj.put("result", result)
        json.put("data",obj)
        json
      }
    }
    case class evaluation(labels:Array[Double],
                          matrixs: Array[Array[Double]],
                          precision:Array[Double],
                          recall:Array[Double],
                          accuracy:Double){
      def make()={
        val json = new JSONObject()
        json.put("labels", labels)
        json.put("matrixs", matrixs)
        json.put("precision", precision)
        json.put("recall", recall)
        json.put("accuracy", accuracy)
        json
      }
    }
    var result = new JSONObject()
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val du = new DataUtils()
    val wr = new WriteResults()
    try {
      require(!(labelCol==null),"label must be set!")
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers, delimiter)
      log.info("================================开始加载模型，进行模型评估！！！")
      val labeledTest = du.toLabeledPoints(sqlContext,orgDataDF, selectCols,labelCol)
      val model = NaiveBayesModel.load(sc, modePath)
      val preLabel =  model.predict(labeledTest.map(_.features))
      val maxProbabis = model.predictProbabilities(labeledTest.map(_.features)).map(_.toArray.max)
      val predictAndLabel = preLabel.zip(labeledTest.map(_.label))
      val strPreLabel = preLabel.zip(maxProbabis).map(x=>(x._1,x._2).toString().replaceAll("\\(","").replaceAll("\\)",""))
      val addColNames = du.getRandomCols(Array("prediction_","score_"):_*)
      val addedData = du.addColumnsToDataFrame(sqlContext,orgDataDF, strPreLabel, addColNames)
      val eva = new Evaluation
      val EVA = eva.getWantedEVA(predictAndLabel, sqlContext)
      val addedAllCols = new ArrayBuffer[String]
      addedAllCols ++= orgDataDF.columns
      addedAllCols ++= addColNames
      result = Reslut("0",null, evaluation(EVA._1, EVA._2, EVA._3, EVA._4, EVA._5).make(),
        sc.applicationId,addedAllCols.toArray ,addColNames(1),addColNames(0), destPath).make()
      log.info("================================开始保存评估结果！！！")
      wr.dataFrameToHdfs(addedData,destPath, delimiter)
    }catch {
      case ex:Exception=>{
        ex.printStackTrace()
        result = Reslut("1", ex.getMessage,null ,null ,null,null,null, null).make()
      }
    }
    log.info("================================开始执行评估结果保存！！！")
    wr.writeResult(resultPath, result.toJSONString)
  }

  /**
    *
    * @param sc: SparkContext
    * @param filePath: 源文件加载路径
    * @param headers: 源文件列名
    * @param selectCols：选择列名
    * @param delimiter：源文件分隔方式
    * @param modePath:模型存放地址
    * @param destPath：结果数据及存放路径
    * @param resultPath：结果集保存路径
    */
  def predict(sc:SparkContext,
              filePath:String,
              headers:Array[String],
              selectCols:Array[String],
              delimiter:String,
              modePath:String,
              destPath:String,
              resultPath:String)={
    case class Reslut(status: String,
                      causeBy:String,
                      types:Array[Double] ,
                      headers: Array[String],
                      predictionColName: String,
                      scoreColName:String,
                      applicationID:String,
                      result:String){
      def make()={
        val json = new JSONObject()
        json.put("status", status)
        json.put("msg", causeBy)
        json.put("applicationID", applicationID)
        val obj = new JSONObject()
        obj.put("types", types)
        obj.put("headers", headers)
        obj.put("predictionColName", predictionColName)
        obj.put("scoreColName", scoreColName)
        obj.put("result", result)
        json.put("data",obj)
        json
      }
    }
    var result = new JSONObject()
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val du = new DataUtils()
    val wr = new WriteResults()
    try{
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers, delimiter)
      val inputDataRDD = du.toVectorRDD( sqlContext,orgDataDF, selectCols)
      log.info("================================开始执行数据预测！！！")
      val model = NaiveBayesModel.load(sc, modePath)
      val predictLabel = model.predict(inputDataRDD)
      val maxProbabis = model.predictProbabilities(inputDataRDD).map(_.toArray.max)
      val addColNames = du.getRandomCols(Array("prediction_","score_"):_*)
      val strPre = predictLabel.zip(maxProbabis).map(x=>(x._1,x._2).toString().replaceAll("\\(","").replaceAll("\\)",""))
      val addPretoData = du.addColumnsToDataFrame(sqlContext, orgDataDF, strPre,addColNames)
      val types = predictLabel.distinct.collect()
      val addedAllCols = new ArrayBuffer[String]
      addedAllCols ++= orgDataDF.columns
      addedAllCols ++= addColNames
      result = Reslut("0",null,types, addedAllCols.toArray, addColNames(0),addColNames(1),sc.applicationId,destPath).make()
      println(result.toJSONString)
      log.info("================================开始保存预测结果！！！")
      wr.dataFrameToHdfs(addPretoData,destPath, delimiter)
    }catch {
      case ex: Exception =>{
        ex.printStackTrace()
        result = Reslut("1", ex.getMessage,null,null ,null,null,null ,null).make()
      }
    }
    wr.writeResult(resultPath, result.toJSONString)
  }
}