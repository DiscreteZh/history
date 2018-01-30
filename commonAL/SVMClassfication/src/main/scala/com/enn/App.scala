package com.enn

import com.alibaba.fastjson.JSONObject
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by enn on 2017/1/3.
  * Author:LYY
  */
object App {
  val log = Logger.getLogger("AppWithSVM")
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
    val testArgs=Array(
      "action=train",
      "taskId=logisiticRegression",
      "resultPathFile=hdfs://node86:8020//user/spark/testDataSet/ttt",
      "modelPathFile=hdfs://node86:8020//user/spark/testDataSet/model",
      "dataSetPathFile=hdfs://node86:8020//user/spark/test/df_X_train.csv",
      "destPath=D:\\Files\\pa.csv",
      "allCols=",
      "labelColName=LABEL",
      "selectCols=DESTINATION_PORT,VOYAGE_NO,DISTRICT_CODE,USD_PRICE,DUTY_VALUE,D_DATE_M,D_DATE_D,D_DATE_W,D_DATE_H,D_DATE_MINUTE",
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
      val resultPath = hdfsRootPath+ params.get("resultPathFile").get
      val modeFileName = hdfsRootPath+ params.get("modelPathFile").get
      val dataSetPathFile = hdfsRootPath+ params.get("dataSetPathFile").get
      var destPath = ""
      if(!params.get("destPath").isEmpty){
        destPath = hdfsRootPath+ params.get("destPath").get
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
    log.info("正在执行SVM算法!")
    val conf = new SparkConf().setAppName(taskId)//submit的标志
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
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val du = new DataUtils()
    val wr = new WriteResults()
    val numIterations = 100
    var result = new JSONObject()
    try{
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers, delimiter)
      val labeledData = du.toLabeledPoints(sqlContext,orgDataDF, selectCols,labelCol)
      log.info("================================开始模型训练！！！")
      val model = SVMWithSGD.train(labeledData, numIterations)
      if ( model != null){
        log.info("================================开始保存训练结果！！！")
        model.save(sc,modePath)
        result = Reslut("0", null, sc.applicationId).make()
        println(result.toJSONString)
      }
    }catch{
      case ex: Exception =>{
        ex.printStackTrace()
        result = Reslut("1",ex.getMessage,null).make()
        println("===========================================================================================")
        println(ex.getMessage)
        println(result.toJSONString)
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
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers, delimiter)
      val labeledTest = du.toLabeledPoints(sqlContext,orgDataDF, selectCols,labelCol)
      log.info("================================开始加载模型，进行模型评估！！！")
      val model = SVMModel.load(sc, modePath)
      val preLabel =  model.predict(labeledTest.map(_.features))
      val addColNames = du.getRandomCols(Array("prediction_","score_"):_*)
      val predictAndLabel = preLabel.zip(labeledTest.map(_.label))
      val strPreLabel = preLabel.zip(preLabel).map(x=>(x._1,x._2).toString().replaceAll("\\(","").replaceAll("\\)","")) // 将zip到一起的两个RDD从Double转化成String类型，并将小括号去掉
      val addedData = du.addColumnsToDataFrame(sqlContext,orgDataDF, strPreLabel, addColNames)
      val eva = new Evaluation()
      val EVA = eva.getWantedEVA(predictAndLabel, sqlContext)
      val addedAllCols = new ArrayBuffer[String]
      addedAllCols ++= orgDataDF.columns
      addedAllCols ++= addColNames
      result = Reslut("0",null, evaluation(EVA._1, EVA._2, EVA._3, EVA._4, EVA._5).make(),
        sc.applicationId,addedAllCols.toArray ,addColNames(1),addColNames(0), destPath).make()
      println(result.toJSONString)
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
                      headers: Array[String],
                      types:Array[Double] ,
                      predictionColName: String,
                      scoreColName: String,
                      applicationID:String,
                      result:String){
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
      val orgDataDF = du.readOrgData(sqlContext, filePath,headers, delimiter)
      orgDataDF.show(100, false)
      val inputDataRDD = du.toVectorRDD( sqlContext,orgDataDF, selectCols)
      log.info("================================开始执行数据预测！！！")
      val model = SVMModel.load(sc, modePath)
      val predictLabel = model.predict(inputDataRDD)
      val addColNames = du.getRandomCols(Array("prediction_","score_"):_*)
      val strPreLabel = predictLabel.zip(predictLabel).map(x=>(x._1,x._2).toString().replaceAll("\\(","").replaceAll("\\)",""))
      val addPretoData = du.addColumnsToDataFrame(sqlContext, orgDataDF, strPreLabel,addColNames)
      val types = predictLabel.distinct.collect()
      val addedAllCols = new ArrayBuffer[String]
      addedAllCols ++= orgDataDF.columns
      addedAllCols ++= addColNames
      result = Reslut("0",null ,addedAllCols.toArray,types, addColNames(0), addColNames(1),sc.applicationId,destPath).make()
      println(result.toJSONString)
      log.info("================================开始保存预测结果！！！")
      wr.dataFrameToHdfs(addPretoData,destPath, delimiter)
    }catch {
      case ex: Exception =>{
        ex.printStackTrace()
        result = Reslut("1", ex.getMessage,null,null ,null, null,null ,null).make()
      }
    }
    wr.writeResult(resultPath, result.toJSONString)
  }
}
