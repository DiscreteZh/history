package com.enn

import com.alibaba.fastjson.JSONObject
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
/**
  * Created by LYY on 2017/1/4.
  */
object App {
  val log = Logger.getLogger("ForestClassification")
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
    * split：原始数据集分割方式
    * coreMax: 最大核数
    */
  def main(args:Array[String]):Unit={
    val TRAIN="train"
    val TEST="test"
    val PREDICTION = "predictions"
    val params = new mutable.HashMap[String,String]()
    val testArgs=Array(
      "action=train",
      "taskId=logisiticRegression",
      "resultPathFile=D:\\Files\\",
      "modelPathFile=hdfs://node86:8020//user/spark/testDataSet/model",
      "dataSetPathFile= D:\\Files\\mulitiClasses.csv",
      "destPath=D:\\Files\\prediction.csv",
      "allCols=A,B,C,D,E",
      "selectCols=A,B,C,D",
      "labelColName=E",
      "split=,",
      "runCoreMax=20")
    testArgs.foreach(x=>{
      val arr = x.split("=")
      if (arr.size > 1) {
        params.put(arr(0), arr(1))
      }
    })
    val action= params.get("action").get
    val taskId = params.get("taskId").get
    val resultPath = params.get("resultPathFile").get
    val modeFileName =  params.get("modelPathFile").get
    val dataSetPathFile =  params.get("dataSetPathFile").get.trim
    var destPath = ""
    if(!params.get("destPath").isEmpty){
      destPath = params.get("destPath").get
    }
   /* val hdfsRootPath = params.get("hdfsRootPath").get
    val action= params.get("action").get
    val taskId = params.get("taskId").get
    val resultPath = hdfsRootPath+ params.get("resultPathFile").get.trim
    val modeFileName = hdfsRootPath+ params.get("modelPathFile").get.trim
    val dataSetPathFile = hdfsRootPath+ params.get("dataSetPathFile").get.trim
    var destPath = ""
    if(!params.get("destPath").isEmpty){
      destPath = hdfsRootPath+ params.get("destPath").get.trim
    }*/
    var labelCol = ""
    if (!params.get("labelColName").isEmpty){
      labelCol = params.get("labelColName").get
    }
    val selectCols = params.get("selectCols").get.split(",")
    var allCols = Array[String]()
    if(!params.get("allCols").isEmpty){
      allCols = params.get("allCols").get.split(",")
    }
    var splitType = new String()
    if(!params.get("split").isEmpty){
      splitType = params.get("split").get
    }else{
      splitType = ","
    }
    val coreMax = params.get("runCoreMax").get
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    log.info("开始执行随机森林分类算法!")
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
    case class Reslut(status: String,causeBy:String,applicationID:String){
      def make()={
        val json = new JSONObject()
        json.put("status", status)
        json.put("msg", causeBy)
        json.put("applicationID", applicationID)
        json
      }
    }
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32
    var result = new JSONObject()
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val du = new DataUtils()
    val wr = new WriteResults()
    try{
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers,delimiter)
      val vectorDF = du.toVectorDataFrame(sqlContext, orgDataDF, selectCols, labelCol)
      val indexedDF = du.makeIndexer(sqlContext,vectorDF, selectCols,labelCol)
      log.info("================================开始模型训练！！！")
      val rf = new RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")
        .setNumTrees(numTrees)
        .setFeatureSubsetStrategy(featureSubsetStrategy)
        .setImpurity(impurity)
        .setMaxDepth(maxDepth)
        .setMaxBins(32)
      val pipeline = new Pipeline()
        .setStages(Array(indexedDF._1, indexedDF._2, rf, indexedDF._3))
      val model = pipeline.fit(vectorDF)
      if ( model != null){
        log.info("================================开始保存模型！！！")
        sc.parallelize(Seq(model), 1).saveAsObjectFile(modePath)
        result = Reslut("0",null, sc.applicationId).make()
        println(result.toJSONString)
      }
    }catch{
      case ex: Exception =>{
        ex.printStackTrace()
        result = Reslut("1",ex.getMessage, null).make()
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
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers,delimiter)
      val vectorDF = du.toVectorDataFrame(sqlContext, orgDataDF, selectCols, labelCol)
      val model = sc.objectFile[PipelineModel](modePath).first()
      val predictions = model.transform(vectorDF)
      val strPrePro = du.getPredictAndProbilities(predictions, sqlContext)
      val addColNames = du.getRandomCols(Array("prediction_","score_"):_*)
      val addedData = du.addColumnsToDataFrame(sqlContext,orgDataDF, strPrePro, addColNames)
      val predictionAndLabels = du.getPredictAndLabel(predictions, sqlContext, labelCol)
      addedData.show(150,false)
      val eva = new Evaluation
      val EVA = eva.getWantedEVA(predictionAndLabels, sqlContext)
      val addedAllCols = new ArrayBuffer[String]
      addedAllCols ++= orgDataDF.columns
      addedAllCols ++= addColNames
      result = Reslut("0",null, evaluation(EVA._1, EVA._2, EVA._3, EVA._4, EVA._5).make(),
        sc.applicationId,addedAllCols.toArray ,addColNames(1),addColNames(0), destPath).make()
      println(result.toJSONString)
      log.info(" 正在执行保存结果数据集到 "+ destPath+" ！！！")
      wr.dataFrameToHdfs(addedData,destPath, delimiter)
    }catch {
      case ex:Exception=>{
        result = Reslut("1",ex.getMessage,null ,null ,null,null,null, null).make()
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
                      headers: Array[String],
                      types:Array[Double] ,
                      predictionColName: String,
                      scoreColName: String,
                      applicationID:String,
                      result:String){ // result:存放数据集的路径destPath
      def make()={
        val json = new JSONObject()
        json.put("status", status)
        json.put("msg", causeBy)
        json.put("applicationID", applicationID)
        val obj = new JSONObject()
        obj.put("headers", headers)
        obj.put("types", types)
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
      val orgDataDF = du.readOrgData(sqlContext, filePath, headers, delimiter)
      val vectorDF = du.toVectorDataFrame(sqlContext, orgDataDF, selectCols)
      val model = sc.objectFile[PipelineModel](modePath).first()
      val predictions = model.transform(vectorDF)
      val addColNames = du.getRandomCols(Array("prediction_","score_"):_*)
      val strPrePro = du.getPredictAndProbilities(predictions, sqlContext)
      val addPretoData = du.addColumnsToDataFrame(sqlContext, orgDataDF, strPrePro,addColNames)
      val types = du.getTypes(predictions, sqlContext)
      val addedAllCols = new ArrayBuffer[String]
      addedAllCols ++= orgDataDF.columns
      addedAllCols ++= addColNames
      result = Reslut("0",null ,addedAllCols.toArray,types, addColNames(0), addColNames(1),sc.applicationId,destPath).make()
      log.info(" 正在执行保存结果数据集到 "+ destPath+" ！！！")
      println(result.toJSONString)
       wr.dataFrameToHdfs(addPretoData,destPath, delimiter)
    }catch {
      case ex: Exception =>{
        result = Reslut("1", ex.getMessage, null, null, null, null, null, null).make()
      }
    }
    wr.writeResult(resultPath, result.toJSONString)
  }
}
