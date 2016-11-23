package data_trans_test

import scala.collection.mutable.WrappedArray
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.log4j.Level
import org.apache.log4j.Logger

object shiyan_tfidf {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("tf-idfTest")
    val spark = new SparkContext(conf)
    val input = "C:/Users/dell/Desktop/data/data_text.txt"
    val src = spark.textFile(input)
    val wordsData = src.map { line =>
      val words = line.split(" ")
      words.toSeq
    }
    val hashingTF = new HashingTF().setHashAlgorithm("native")
    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.repartition(1).saveAsTextFile("C:/Users/dell/Desktop/out2")
    val newidf = new IDF()
    val NewidfModel = newidf.fit(featurizedData)
    val vecTFIDF = NewidfModel.transform(featurizedData)
    vecTFIDF.repartition(1).saveAsTextFile("C:/Users/dell/Desktop/out3")
    val wordsarr = wordsData.map { line =>
      val lineword = line.toArray
      lineword
    }
    val arrrdd = wordsarr.collect()
    val ondis = arrrdd(0) ++ arrrdd(1) ++ arrrdd(2) ++ arrrdd(3) ++ arrrdd(4) ++ arrrdd(5)
    val dis = ondis.distinct
    dis.foreach { println }
    val hot = vecTFIDF.map {
      case SparseVector(size, indices, values) =>
        var hashnumarr = indices
        var tfidfvalues = values
        val arrbuf = new ArrayBuffer[String]
        for (i <- 1 to 3) {
          //寻找每个向量中tf-idf最大的3个词语
          val findmax = tfidfvalues.indexOf(tfidfvalues.max)
          //找到最大值后至0
          tfidfvalues(findmax) = 0.0
          val maxhashnum = hashnumarr(findmax)
          for (word <- dis) {
            if (word.hashCode() == maxhashnum)
              arrbuf += word
          }
        }
        arrbuf

    }

    hot.repartition(1).saveAsTextFile("C:/Users/dell/Desktop/hotest")

  }
}