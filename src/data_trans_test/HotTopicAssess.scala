package data_trans_test

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark._
import org.apache.spark.sql._
import org.apache.log4j.Level
import org.apache.log4j.Logger
import java.text.SimpleDateFormat
import java.util.Date

object HotTopicAssess {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("HotTopicAssess")
    val spark = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    //输入路径
    val inputpath = "C:/Users/dell/Desktop/clustervec/clustervec0"
    //输出路径
    val outputpath = "C:/Users/dell/Desktop/HotAssessResult/"
    val DS = spark.textFile(inputpath)
    DS.cache
    //时间戳处理

    //获取当前的时间戳
    def getNowDate(): String = {
      var now: Date = new Date()
      var dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd,HH:mm")
      var nowtime = dateFormat.format(now)
      nowtime
    }
    //计算和当前时间戳的时间差(/秒)
    def caculateTime(start_time: String, end_time: String): Long = {
      var df: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd,HH:mm")
      var begin: Date = df.parse(start_time)
      var end: Date = df.parse(end_time)
      var between: Long = (end.getTime() - begin.getTime()) / 1000
      between
    }
    //计算帖子的时效性
    val caculaTimeDS = DS.map {
      line =>
        val split = line.split(" ")
        //index,topicDistribution,maxprobability,prediction,time,recall
        val timestamp = caculateTime(split(4).toString, getNowDate())
        (split(0).toInt, split(1).toString(), split(2).toDouble, split(3).toInt, timestamp, split(5).toInt)
    }
    caculaTimeDS.cache
    caculaTimeDS.repartition(1).saveAsTextFile(outputpath + "caculaTimeDS")
    //2016-12-29
    val rawData = caculaTimeDS.map {
      case (index, topicDistribution, maxprobability, prediction, time, recall) =>

    }

  }
}