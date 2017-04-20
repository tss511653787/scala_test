package HotalDataHotWords
//import
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.sql._
import utils.AnaylyzerTools
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.log4j.Level
import org.apache.log4j.Logger
import java.io.PrintWriter
import java.util.ArrayList
import scala.collection.mutable.ArrayBuffer
import data_trans_test._

class participle {

}
object participle {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("Hoteldata")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //引入隐式转换
    import sqlContext.implicits._
    //文件路径 
    val path = "C:/Users/Administrator/Desktop/wanghao/wanhaodata_d.csv"
    val newpath = "C:/Users/Administrator/Desktop/ouput/"

    //读取文件
    val text = sc.textFile(path)
    text.cache()
    val data = text.map {
      rawline =>
        //csv格式按照，分割
        val split = rawline.split(",")
        DataRecord(split(0).toDouble, split(2), split(1))
    }
    val dataDF = data.toDF();
    dataDF.cache();
    //分词处理
    val casetext = dataDF.rdd.map {
      case Row(score: Double, comments: String, customer_type: String) =>
        comments
    }
    //为每个条数建立索引序号(1)
    var indexnum = 0
    val caseall = dataDF.rdd.map {
      case Row(score: Double, comments: String, customer_type: String) =>
        indexnum = indexnum + 1
        (indexnum, score, comments, customer_type)
    }
    val caseallDF = caseall.toDF
    println("标签条数" + caseallDF.count())
    caseallDF.cache
    caseallDF.show
    //重置列名属性
    val recaseDF = caseallDF
      .select(caseallDF("_1").as("index"), caseallDF("_2").as("score"), caseallDF("_3").as("comments"), caseallDF("_4").as("customer_type"))
    recaseDF.cache
    recaseDF.show()
    //数据落地
    val arrtext = casetext.collect
    val savechar = new PrintWriter(newpath + "savechar")
    arrtext.foreach { line => savechar.println(line) }
    savechar.close()

    //分词
    //运行分词类
    CopyOfAnaylyzerTools.split()
    //再次读入结果
    val inputpath = "C:/Users/Administrator/Desktop/splitoutput/*"
    val wordsDS = sc.wholeTextFiles(inputpath)
    wordsDS.cache
    val toint = wordsDS.map {
      case (file, text) => (file.split("/")(6).toInt, text)
    }
    toint.cache
    val renamedata = toint.map {
      line =>
        record(line._1, line._2)
    }
    val tointDF = renamedata.toDF()
    tointDF.cache
    tointDF.rdd.repartition(1).saveAsTextFile(newpath + "tointDF")
    tointDF.show

    //将持久化的分词结果和元数据进行笛卡尔积
    // val datajoinDF = tointDF.join(recaseDF, tointDF("index") === recaseDF("index"))
    val datajoinDF = tointDF.join(recaseDF, "index")
    datajoinDF.cache
    val DataDF = datajoinDF
    DataDF.cache
    println("DataDF:")
    DataDF.show
    DataDF.rdd.repartition(1).saveAsTextFile(newpath + "dataDF")
  }

}
case class DataRecord(score: Double, comments: String, Customer_type: String)
case class record(index: Int, splitwords: String)