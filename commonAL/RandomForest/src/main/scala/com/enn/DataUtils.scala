package com.enn

import com.enn.App.log
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import scala.collection.mutable.ArrayBuffer

class DataUtils {

  /**
    *
    * @param sqlContext : SQLContext
    * @param sourseDataPath: 加载源文件路径
    * @param allClumns：源文件对应的所有字段名称集合
    * @param dataDelimiter：源文件数据分割方式
    * @return ：源文件DataFrame
    */
  def readOrgData(sqlContext: SQLContext,
                  sourseDataPath:String,
                  allClumns:Array[String],
                  dataDelimiter:String): DataFrame={
    log.info("正在加载原始数据。。。。。。")
    if (allClumns.length > 0){
      val loadData: DataFrame = sqlContext.read
        .format("com.databricks.spark.csv")
        .option("header", "false")
        .option("inferSchema", "false")
        .option("delimiter",dataDelimiter)
        .load(sourseDataPath)
      val orgData = loadData.toDF(allClumns:_*)
      return orgData
    } else{
      val orgData: DataFrame = sqlContext.read
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("delimiter",dataDelimiter)
        .load(sourseDataPath)
      return orgData
    }
  }
  def makeIndexer(sqlContext:SQLContext,
                  inputData:DataFrame,
                  selectCols:Array[String],
                  labelCol:String*)={
    val labelIndexer = new StringIndexer()
      .setInputCol(labelCol(0))
      .setOutputCol("indexedLabel")
      .fit(inputData)
    inputData.show(10, false)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(inputData)
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictionLabel")
      .setLabels(labelIndexer.labels)
    (labelIndexer, featureIndexer, labelConverter)
  }
  /**
    *
    * @param sqlContext:SQLContext
    * @param inputDF:输入数据集（DataFrame）
    * @param selectCols:选择用于参与训练的特征列
    * @param labelCols：标签列
    * @return DataFrame(label, features)
    */
  def toVectorDataFrame(sqlContext:SQLContext,
                        inputDF:DataFrame,
                        selectCols:Array[String],
                        labelCols:String*):DataFrame={
    log.info("从原始数据中选择指定列。。。")
    inputDF.registerTempTable("org_data")
    sqlContext.udf.register("getDouble",(i:String)=>{if(i.trim.isEmpty) 0.0f else i.toDouble})
    val getherCols = new ArrayBuffer[String]
    getherCols ++= selectCols
    if (labelCols.length > 0){
      getherCols += labelCols(0)
      val colString = getherCols.mkString("),getDouble(" )
      val selectDF = sqlContext.sql("select getDouble("+ colString+") from org_data")
      val newDF =selectDF .toDF(getherCols.toArray: _*)
      val assembler = new VectorAssembler()
        .setInputCols(selectCols)
        .setOutputCol("features")
      val vectorDF= assembler.transform(newDF).select(labelCols(0),"features")
      return vectorDF
    }else{
      val colString = getherCols.mkString("),getDouble(" )
      val selectDF = sqlContext.sql("select getDouble("+ colString+") from org_data")
      val newDF =selectDF .toDF(getherCols.toArray: _*)
      val assembler = new VectorAssembler()
        .setInputCols(selectCols)
        .setOutputCol("features")
      val vectorDF= assembler.transform(newDF).select("features")
      return vectorDF
    }
  }
  /**
    * @param sqlContext: SQLContext
    * @param inputDF: 读入的原始数据
    * @param addedRDD：需要添加的数据列
    * @param addColNames：添加列的列名
    * @return DataFrame
    */
  def addColumnsToDataFrame(sqlContext: SQLContext,
                            inputDF:DataFrame,
                            addedRDD:RDD[String],
                            addColNames:Array[String]) ={
    log.info("================================开始添加需要的列到原始数据！！！")
    val resultRDD = inputDF.rdd.map(x=>x.toString().replaceAll("\\[","").replaceAll("\\]","")).zip(addedRDD)
    val strResultRDD = resultRDD.map(x=>(x._1,x._2).toString().replaceAll("\\(","").replaceAll("\\)",""))
    val inputCols = inputDF.columns
    val Rowrdd = strResultRDD.map(x=>Row.fromSeq(x.split(",").toSeq))
    val RowCols = new ArrayBuffer[String]()
    RowCols ++= inputDF.columns
    RowCols ++= addColNames
//    val colArray: Array[String] = Array.range(0,RowCols.size).map(i => (i.toString))
    val structType = StructType( RowCols.map(fieldName => StructField(fieldName, StringType, true)))
    sqlContext.createDataFrame(Rowrdd, structType)
//    sqlRDD.toDF(RowCols:_*)
  }
  def getLabelType(label:RDD[Double]):Int={
    val types = label.distinct.count().toInt
    return types
  }

  /**
    * 给定字符串添加随机数
    * @param col:输入字符串集
    * @return
    */
  def getRandomCols(col:String*):Array[String]={
    val rnd = new scala.util.Random
    val colNames = new ArrayBuffer[String]()
    for (i<- col){
      colNames += i+ 1 + rnd.nextInt( (1000 - 1) + 1 ).toString
    }
    return colNames.toArray
  }

  def getPredictAndProbilities(predictions:DataFrame,
                               sqlContext: SQLContext)={
    predictions.registerTempTable("results")
    val prediction = sqlContext.sql("select predictionLabel from results ")
    val probilities = sqlContext.sql("select probability from results ")
    val MaxPro = probilities.map{ row =>
      row.getAs[Vector](0).toArray.max
    }
    val strPre = prediction.rdd.map(x=>x.toString().replaceAll("\\[","").replaceAll("\\]",""))
    strPre.zip(MaxPro).map(x=>(x._1,x._2).toString().replaceAll("\\(","").replaceAll("\\)",""))
  }
  def getPredictAndLabel(predictions:DataFrame,
                         sqlContext: SQLContext,
                         labelCol:String)={
    predictions.registerTempTable("results")
    val label = sqlContext.sql("select "+ labelCol+" from results ")
    val prediction = sqlContext.sql("select predictionLabel from results ")
    val strPre = prediction.rdd.map(x=>x.toString().replaceAll("\\[","").replaceAll("\\]",""))
    val strLabel = label.rdd.map(x=>x.toString().replaceAll("\\[","").replaceAll("\\]",""))
    strPre.zip(strLabel).map(x=>(x._1.toDouble,x._2.toDouble))
  }
  def getTypes(predictions:DataFrame, sqlContext: SQLContext):Array[Double]={
    predictions.registerTempTable("results")
    val prediction = sqlContext.sql("select predictionLabel from results ")
    val strPre = prediction.rdd.map(x=>x.toString().replaceAll("\\[","").replaceAll("\\]",""))
    val types = strPre.map(x=>x.toDouble).distinct().collect()
    return types
  }
}
