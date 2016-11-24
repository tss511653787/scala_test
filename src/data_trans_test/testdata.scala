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

object testdata {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("testdata")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val path = "C:/Users/dell/Desktop/data/Cardata250.csv"
    //path:hdfs://tss.hadoop2-1:8020/user/root/dataSet/data/test2_bsk.csv C:/Users/dell/Desktop/data/Cardata250.csv
    //读取文件
    val text = sc.textFile(path)
    text.cache()
    //获取列数据
    val data = text.map {
      rawline =>
        val splitdata = rawline.split(",")
        DataRecord(splitdata(0), splitdata(1), splitdata(2), splitdata(3), splitdata(4), splitdata(5).toInt, splitdata(6))
    }
    val dataDF = data.toDF
    dataDF.show
    //保存dataDF
    val newpath = "C:/Users/dell/Desktop/ouput/"
    dataDF.rdd.repartition(1).saveAsTextFile(newpath + "dataDF")
    //分词处理
    val list = dataDF.rdd.map {
      case Row(title: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
      //        val splitword = AnaylyzerTools.anaylyzerWords(text)
      //        System.runFinalization()
      //        val arrword = splitword.toArray()
      //        splitword.toArray()

    }
    list.repartition(1).saveAsTextFile(newpath + "list")
    //    user_text.repartition(500)
    //    val fenci = user_text.map(
    //      line => {
    //        //内存不足错误 GC 
    //        //需完善
    //        val list = AnaylyzerTools.anaylyzerWords(line) //按行进行map分词 结果返回Arraylist
    //        list.toString()
    //          .replace("[", "").replace("]", "")
    //          .replaceAll(" ", "").replaceAll(",", " ")
    //          .replaceAll("[(]", "").replaceAll("[)]", "")
    //      })
    //    fenci.cache()
    //    //统计单词
    //    val num = fenci.flatMap(_.split(" "))
    //    val number = num.distinct().count()
    //    println("分词去重统计：" + number)
    //    //100行大概有6626的不重复单词
    //    //初步将： 分词结果 ，发帖时间 ， 回复数 作为输入数据
    //    val inputData = fenci
    //      .zip(user_time)
    //      .zip(user_recall)
    //      .zipWithIndex
    //
    //    val inputformat = inputData.map(line => {
    //      val temp = line.toString.replaceAll("[(]", "").replaceAll("[)]", "")
    //      temp
    //    })
    //    //位置整理
    //    val input = inputformat.map {
    //      line =>
    //        val index = line.split(",")(3)
    //        val text = line.split(",")(0)
    //        val time = line.split(",")(1)
    //        val recall = line.split(",")(2)
    //        (index, text, time, recall)
    //    }
    //    val inputDataSet = input.map(line => {
    //      val temp = line.toString.replaceAll("[(]", "").replaceAll("[)]", "")
    //      temp
    //    })
    //    inputDataSet.repartition(1).saveAsTextFile(newpath + "inputdata")

    //正则过滤  还在进行
    //    val regex = """[^0-9]*""".r
    //    val fenci_filter = fenci.map { line =>
    //      line.split(", ").filter(token => regex.pattern.matcher(token).matches)
    //    }

  }
}
case class DataRecord(title: String, content: String, text: String, name: String, time: String, recall: Int, Net: String)
