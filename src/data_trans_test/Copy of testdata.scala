package data_trans_test

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


object Copy_testdata {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("testdata")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val path = "C:/Users/dell/Desktop/data/Cardata50.csv"
    //path:hdfs://tss.hadoop2-1:8020/user/root/dataSet/data/test2_bsk.csv C:/Users/dell/Desktop/data/Cardata250.csv
    //读取文件
    val text = sc.textFile(path)
    text.cache()
    //获取列数据
    val data = text.map {
      rawline =>
        val splitdata = rawline.split(",")
        CopyDataRecord(splitdata(0), splitdata(1), splitdata(2), splitdata(3), splitdata(4), splitdata(5).toInt, splitdata(6))
    }
    val dataDF = data.toDF
    dataDF.cache
    dataDF.show
    //保存dataDF
    val newpath = "C:/Users/dell/Desktop/ouput/"
    dataDF.rdd.repartition(1).saveAsTextFile(newpath + "dataDF")
    //分词处理
    val casetext = dataDF.rdd.map {
      case Row(title: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
        text
    }
    var indexnum = 0
    val caseall = dataDF.rdd.map {
      case Row(title: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
        indexnum = indexnum + 1
        (indexnum, title, content, text, name, time, recall, net)
    }
    val caseallDF = caseall.toDF
    caseallDF.show
    caseallDF.cache
    val recaseDF = caseallDF.select(caseallDF("_1").as("index"), caseallDF("_2").as("title"), caseallDF("_3").as("content"), caseallDF("_4").as("text"), caseallDF("_5").as("name"), caseallDF("_6").as("time"), caseallDF("_7").as("recall"), caseallDF("_8").as("net"))
    recaseDF.cache
    val arrtext = casetext.collect
    val savechar = new PrintWriter(newpath + "savechar")
    arrtext.foreach { line => savechar.println(line) }
    //使用javaIO输出必须关闭流否则会被释放掉 切记！！！
    savechar.close
    //运行分词类
    CopyOfAnaylyzerTools.split()
    //再次读入结果
    val inputpath = "C:/Users/dell/Desktop/splitoutput/*"
    val wordsDS = sc.wholeTextFiles(inputpath)
    wordsDS.cache
    val toint = wordsDS.map {
      case (file, text) => (file.split("/")(6).toInt, text)
    }
    toint.cache
    val tointDF = toint.toDF()
    tointDF.cache
    tointDF.show
    val renamedata = tointDF.select(tointDF("_1").as("index"), tointDF("_2").as("splitwords"))
    renamedata.cache
    //连接的笛卡尔积运算 计算量很大
    val datajoinDF = renamedata.join(recaseDF, renamedata("index") === recaseDF("index"))
    datajoinDF.cache
    val DataDF = datajoinDF
    DataDF.cache
    DataDF.show
    DataDF.rdd.repartition(1).saveAsTextFile(newpath + "ddataDF")
    /* 由于无法序列化分词包中的方法目前分词暂时无法集成在Spark中处理
        val anaylyzerTools = new CopyOfAnaylyzerTools()
        val configg = anaylyzerTools.config
        val dicc = anaylyzerTools.dic
        val splitText = casetext.map {
          line =>
            val list = anaylyzerTools.anaylyzerWords(line, configg, dicc)
            list.toString()
              .replace("[", "").replace("]", "")
              .replaceAll(" ", "").replaceAll(",", " ")
              .replaceAll("[(]", "").replaceAll("[)]", "")
        }
        splitText.repartition(1).saveAsTextFile(newpath + "splitText")
        */
    DataDF.explain()
    //单词去重统计
    val spword = DataDF.rdd.map {
      case Row(index: Int, splitwords: String, aindex: Int, titile: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
        (index, splitwords, time, recall)
    }
    //spword RDD
    //格式:[index,分词，时间，回复数]
    spword.cache()
    spword.repartition(1).saveAsTextFile(newpath + "spword")
    val numword = spword.map {
      case (index, splitwords, time, recall) =>
        splitwords
    }
    numword.cache
    val num = numword.flatMap(_.split(" "))
    val number = num.distinct().count()
    println("分词去重统计：" + number)
    val outdata = spword.map(line => {
      val str = line.toString
      val reback = str.replace("[", "").replace("]", "").replace("[(]", "").replace("[)]", "")
      reback
    })
    outdata.cache
    outdata.repartition(1).saveAsTextFile(newpath + "outdata")
    //rdd存储下来有()使用notepad替换掉即可
    //然后在转换成utf-8格式

    //正则过滤  还在进行
    //    val regex = """[^0-9]*""".r
    //    val fenci_filter = fenci.map { line =>
    //      line.split(", ").filter(token => regex.pattern.matcher(token).matches)
    //    }

  }
}
case class CopyDataRecord(title: String, content: String, text: String, name: String, time: String, recall: Int, Net: String)
