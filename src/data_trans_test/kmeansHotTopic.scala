package data_trans_test

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import scala.reflect.runtime.universe
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.clustering.KMeansModel
import java.io.PrintWriter

object kmeansFindPaperHotTopic {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("kmeans-test")
    val spark = new SparkContext(conf)
    //引入spark sql标签
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    //读取数据
    val inputpath = "C:/Users/dell/Desktop/data/"
    val src = spark.textFile(inputpath + "kmeans")
    val srcDS = src.map {
      line =>
        var data = line.split(",")
        RawDataRecord(data(0), data(1), data(2).toInt)
    }

    //70%作为训练数据，30%作为测试数据
    //    val splits = srcDS.randomSplit(Array(0.7, 0.3))
    //将打标签的数据转换成DataFrame数据格式
    val trainingDF = srcDS.toDF()
    //    val testDF = src.toDF()

    //将词语转换成数据,tokenizer是按着空格切分
    val rextokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val wordsData = rextokenizer.transform(trainingDF)
    println("text+words:")
    //观察结果
    wordsData.cache
    wordsData.show
    val newpath = "C:/Users/dell/Desktop/result/"
    // wordsData.rdd.repartition(1).saveAsTextFile(newpath+"text_words")

    //计算每个词在文档中的词频
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol("words")
      .setOutputCol("hashWithTf")
    val featurizedData = hashingTF.transform(wordsData)
    println("text+words+hashWithTf:")
    featurizedData.cache
    featurizedData.show
    featurizedData.rdd.repartition(1).saveAsTextFile(newpath + "text_words_hashWithTf")
    featurizedData.select($"hashWithTf").take(9).foreach(println)

    //计算每个词的IDF
    val idf = new IDF().setInputCol("hashWithTf").setOutputCol("features")
    //计算每个词的TF-IDF
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.cache
    rescaledData.show
    rescaledData.rdd.repartition(1).saveAsTextFile(newpath + "text_words_hashWithTf_idfnum")

    //k-means
    val kmeans = new KMeans()
      .setK(5)
      .setSeed(1L)
      //使用计算好的tf-idf值作为输入
      //.setFeaturesCol("idfnum")
    //.setPredictionCol("predictions")
    //fitModel
    val kmeansModel = kmeans.fit(rescaledData)
    val clusterCenters = kmeansModel.clusterCenters;
    //聚类中心
    println("聚类中心:")
    clusterCenters.foreach(println)
    val linenum = clusterCenters.length
    val clusterCenters_output = new PrintWriter(newpath + "clusterCenters_output")
    for (line <- 0 to linenum - 1) {
      clusterCenters_output.println(clusterCenters(line).toString)
    }
    clusterCenters_output.close

    //误差计算
    val WSSSE = kmeansModel.computeCost(rescaledData)
    println("with set sum of squared errors:" + WSSSE)

    val predictioned = kmeansModel.summary.predictions;

    //取前n行的聚类结果指定列输出到控制台
    kmeansModel.summary.predictions
      .select($"prediction", $"text")
      .take(9)
      .foreach { println }
    kmeansModel.summary.predictions.show(50)

    //保存带有聚类了类簇标签的文件到指定文件夹
    kmeansModel.summary.predictions
      .rdd
      .repartition(1)
      .saveAsTextFile(newpath + "kmeans-summary")

  }
  case class RawDataRecord(text: String, time: String, recall: Int)
}


























































