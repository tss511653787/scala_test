package data_trans_test

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.sql._
import utils.AnaylyzerTools
import org.apache.spark.mllib.regression.LabeledPoint

object testdata {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName("testdata")

    val sc = new SparkContext(conf)
    val path = "C:/Users/dell/Desktop/data/Cardata50.csv"
    //path:hdfs://tss.hadoop2-1:8020/user/root/dataSet/data/test2_bsk.csv
    //读取文件
    val text = sc.textFile(path)
    text.cache()
    //获取列数据
    val user_title = text.map(_.split(",")(0))
    val user_content = text.map(_.split(",")(1))
    val user_text = text.map(_.split(",")(2))
    val user_name = text.map(_.split(",")(3))
    val user_time = text.map(_.split(",")(4))
    val user_recall = text.map(_.split(",")(5))
    val user_net = text.map(_.split(",")(6))

    //测试列级提取
    val newpath = "C:/Users/dell/Desktop/ouput/"
    user_text.cache
    user_text.repartition(1).saveAsTextFile(newpath + "inputtext")
    user_text.cache
    //分词处理
    val fenci = user_text.map(
      x => {
        val list = AnaylyzerTools.anaylyzerWords(x) //按行进行map分词 结果返回Arraylist
        list.toString()
          .replace("[", "").replace("]", "")
          .replaceAll(" ", "").replaceAll(",", " ")
          .replaceAll("[(]", "").replaceAll("[)]", "")

      })
    fenci.cache()
    //统计单词
    val num = fenci.flatMap(_.split(" "))
    val number = num.distinct().count()
    println("分词去重统计：" + number) //100行大概有6626的不重复单词
    //初步将： 分词结果 ，发帖时间 ， 回复数 作为输入数据
    val inputData = fenci
      .zip(user_time)
      .zip(user_recall)
      .zipWithIndex

    val inputformat = inputData.map(line => {
      val temp = line.toString.replaceAll("[(]", "").replaceAll("[)]", "")
      temp
    })
    //位置整理
    val input = inputformat.map {
      line =>
        val index = line.split(",")(3)
        val text = line.split(",")(0)
        val time = line.split(",")(1)
        val recall = line.split(",")(2)
        (index, text, time, recall)
    }
    val inputDataSet = input.map(line => {
      val temp = line.toString.replaceAll("[(]", "").replaceAll("[)]", "")
      temp
    })
    inputDataSet.repartition(1).saveAsTextFile(newpath + "inputdata")

    //正则过滤  还在进行。。。
    //    val regex = """[^0-9]*""".r
    //    val fenci_filter = fenci.map { line =>
    //      line.split(", ").filter(token => regex.pattern.matcher(token).matches)
    //    }

  }
}
