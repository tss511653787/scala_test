package data_trans_test

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.linalg.Vector


object shiyan_count {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("tf-idfTest")
    val spark = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    val input = "C:/Users/dell/Desktop/data/CountVecinput.txt"
    val src = spark.textFile(input)

    //    val srcDS = src.map {
    //      line =>
    //        var data = line.split(" ")
    //        countDataRecord(data(0))
    //    }
    val countDF = src.toDF()
    countDF.show()
    val rextokenizer = new Tokenizer()
      .setInputCol("value")
      .setOutputCol("words")
    val wordsData = rextokenizer.transform(countDF)
    wordsData.show
    val countvectorizer = new CountVectorizer()
    countvectorizer.setInputCol("words").setOutputCol("vec")
    val countvectorizermodel = countvectorizer.fit(wordsData)
    val LDAWithvec = countvectorizermodel.transform(wordsData)
    LDAWithvec.show
    val outputpath1 = "C:/Users/dell/Desktop/"
    //LDAWithvec.rdd.repartition(1).saveAsTextFile(outputpath1 + "LDAWithvec")
    val vec = LDAWithvec.rdd.map{
      case Row(value:String,words:WrappedArray[String],vec:Vector)=>
        (vec.toDense)
    }
    vec.repartition(1).saveAsTextFile(outputpath1+"Dencevec")
    

  }

}
case class countDataRecord(text: String)