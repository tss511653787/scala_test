package data_trans_test

import java.text.SimpleDateFormat
import java.util.Date

import scala.math._

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark._
import org.apache.spark.sql._

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
        //设置时间戳:当前时间-发帖时间
        val timestamp = caculateTime(split(4).toString, getNowDate())
        (split(0).toInt, split(1), split(2).toDouble, split(3).toInt, timestamp, split(5).toInt)
    }
    caculaTimeDS.cache
    caculaTimeDS.repartition(1).saveAsTextFile(outputpath + "caculaTimeDS")
    //为数据打标签
    val rawData = caculaTimeDS.map {
      case (index, topicDistribution, maxprobability, prediction, time, recall) =>
        AccessDataRecord(index, topicDistribution, maxprobability, prediction, time, recall)
    }
    rawData.toDF().show
    /*
     * 计算话题话题热度指标
     * 帖子数
     * 关注度
     * 时效性
     * 突发度
     * 纯净度
     */
    //聚簇中帖子数量
    val eleNum = caculaTimeDS.count.toInt

    //关注度计算
    var attenDeg = 0.0
    val recallRDD = caculaTimeDS.map {
      case (index, topicDistribution, maxprobability, prediction, time, recall) =>
        recall
    }
    //recallArr顺序发生变化
    val recallArr = recallRDD.collect
    val logValuearr = new Array[Double](eleNum)
    for (i <- 0 until eleNum) {
      logValuearr(i) = log10(recallArr(i) + 1) / log10(2)
    }
    attenDeg = logValuearr.sum / eleNum

    //时效性计算
    var timeDeg = 0.0
    val timeRDD = caculaTimeDS.map {
      case (index, topicDistribution, maxprobability, prediction, time, recall) =>
        time
    }
    //timeArr顺序发生变化
    val timeArr = timeRDD.collect
    val timeValue = new Array[Double](eleNum)
    for (i <- 0 until eleNum) {
      timeValue(i) = log10(timeArr(i) + 1) / log10(2)
    }

  }
}
