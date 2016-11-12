package data_trans_test
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import java.io.PrintWriter
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.Tokenizer

object LDAHotTopic {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("LDA-test")
    val spark = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    val inputpath = "C:/Users/dell/Desktop/data/"
    val outputpath = "C:/Users/dell/Desktop/LDAresult/"
    val src = spark.textFile(inputpath + "kmeans1")
    val srcDS = src.map {
      line =>
        var data = line.split(",")
        RawDataRecord(data(0).toInt, data(1), data(2), data(3).toInt)
    }
    val LDAinputDF = srcDS.toDF()
    LDAinputDF.show()
    //将text按空格切分转化为数组 WrappedArray
    val rextokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val wordsData = rextokenizer.transform(LDAinputDF)
    wordsData.show
    //
    val rddtext = LDAinputDF.rdd.map {
      case Row(index: Int, text: String, time: String, recall: Int) =>
        (text)
    }
    //
    val countvectorizer = new CountVectorizer()
    countvectorizer.setInputCol("words").setOutputCol("LDAvec")
    val countvectorizermodel = countvectorizer.fit(wordsData)
    val LDAWithvec = countvectorizermodel.transform(wordsData)
    LDAWithvec.show
    LDAWithvec.rdd.repartition(1).saveAsTextFile(outputpath+"LDAWithvec")

  }
}