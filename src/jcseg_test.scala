

import org.apache.spark._
import utils.AnaylyzerTools
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

/**
 * Created by Administrator on 2016/4/6.
 */
object jcseg_test {

  //分词排序后取出词频最高的前10个
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("my app").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val data = sc.textFile("C:/Users/dell/Desktop/test.txt").map(x => {
      val list = AnaylyzerTools.anaylyzerWords(x) //分词处理
      list.toString.replace("”", "").replace("”", "").split(",")
    }).flatMap(x => x.toList)
    .map(x => (x.trim(), 1))
    .reduceByKey(_ + _)
    .collect.sortBy(-_._2)
    .mkString("\n").foreach(print)
  }

 
}