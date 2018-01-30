package com.enn

import java.io.{File, PrintWriter}
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.DataFrame

class WriteResults {
  /**
    * @param inputData: 输入数据集
    * @param path：数据集存放路径
    * @param delimiter:保存数据集的分隔方式
    */
  def dataFrameToHdfs(inputData:DataFrame, path:String,delimiter:String )={
    inputData.write.format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter", delimiter)
      .save(path)
  }
  /**
    *
    * @param resultPath: 文件保存路径
    * @param result： 要保存的文件（JSONString）
    */
  def writeResult (resultPath: String, result: String )={
    try{
      println("正在向hdfs写文件")
      val filePath = resultPath + "/" + "result"
      val hdfs = FileSystem.get(URI.create(resultPath),new Configuration())
      var fsDataOutputStream = hdfs.create(new Path(filePath))
      val writer = new PrintWriter(fsDataOutputStream)
      writer.write(result)
      writer.close()
    }catch {
      case ex: Exception =>{
        ex.printStackTrace()
      }
    }
  }
}
