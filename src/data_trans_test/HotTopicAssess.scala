package data_trans_test

import java.text.SimpleDateFormat
import java.util.Date

import scala.math._

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark._
import org.apache.spark.sql._
import java.io.PrintWriter

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
        val timestamp = caculateTime(split(5).toString, getNowDate())
        (split(0).toInt, split(1).toInt, split(2), split(3).toDouble, split(4).toInt, timestamp, split(6).toInt)
    }
    caculaTimeDS.cache
    caculaTimeDS.repartition(1).saveAsTextFile(outputpath + "caculaTimeDS")
    //为数据打标签
    val rawData = caculaTimeDS.map {
      case (postNum, index, topicDistribution, maxprobability, prediction, time, recall) =>
        AccessDataRecord(postNum, index, topicDistribution, maxprobability, prediction, time, recall)
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
    //帖子比重
    var eleNumDeg = 0.0
    val numberRDD = caculaTimeDS.map {
      case (postNum, index, topicDistribution, maxprobability, prediction, time, recall) =>
        postNum
    }
    val postNumberArr = numberRDD.collect.toArray
    val postNumber = postNumberArr(0)
    eleNumDeg = postNumber.toDouble / (4958).toDouble

    //关注度计算
    var attenDeg = 0.0
    val recallRDD = caculaTimeDS.map {
      case (postNum, index, topicDistribution, maxprobability, prediction, time, recall) =>
        recall
    }
    //recallArr顺序发生变化
    val recallArr = recallRDD.collect
    val logValuearr = new Array[Double](eleNum)
    for (i <- 0 until eleNum) {
      logValuearr(i) = log10(recallArr(i) + 1) / log10(20)
    }
    attenDeg = logValuearr.sum / eleNum

    //时效性计算
    var timeDeg = 0.0
    val timeRDD = caculaTimeDS.map {
      case (postNum, index, topicDistribution, maxprobability, prediction, time, recall) =>
        time
    }
    //timeArr顺序发生变化
    val timeArr = timeRDD.collect
    val timeValue = new Array[Double](eleNum)
    for (i <- 0 until eleNum) {
      timeValue(i) = log10(timeArr(i) + 1) / log10(6000)
    }
    timeDeg = (timeValue.sum / eleNum) * (-1)

    //突发度计算
    //整个聚类的突发度指标pd
    var pd = 0.0
    var promDeg = 0.0
    //每个“主题”的突发度
    val promRDD = caculaTimeDS.map {
      case (postNum, index, topicDistribution, maxprobability, prediction, time, recall) =>
        maxprobability
    }
    val promArr = promRDD.collect
    val average = promArr.sum / eleNum
    for (i <- 0 until eleNum) {
      promDeg += pow(promArr(i) - average, 2)
    }
    promDeg = promDeg / eleNum
    for (i <- 0 until eleNum) {
      pd += promDeg * promArr(i)
    }

    //纯净度计算
    var prueDeg = 0.0
    var denomin = 0.0
    for (i <- 0 until eleNum) {
      denomin += (log10(promArr(i)) / log10(2)) * promArr(i)
    }
    prueDeg = (1 / denomin) * (-1)

    //热度
    val hotDeg = eleNumDeg + attenDeg + timeDeg + pd + prueDeg

    //记录保存
    val indicatorRes = new PrintWriter(outputpath + "indicatorRes")
    indicatorRes.println("输出结果如下:")
    indicatorRes.println("帖子数量:" + postNumber)
    indicatorRes.println("帖子比重:" + eleNumDeg)
    indicatorRes.println("关注度:" + attenDeg)
    indicatorRes.println("时效性:" + timeDeg)
    indicatorRes.println("突发度:" + pd)
    indicatorRes.println("纯净度:" + prueDeg)
    indicatorRes.println("热度:" + hotDeg)
    indicatorRes.close

  }
}
