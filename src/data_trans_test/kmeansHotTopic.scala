package data_trans_test

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{ Vector => mllibVector }
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import scala.reflect.runtime.universe
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.io.PrintWriter
import breeze.linalg._
import breeze.numerics.pow
import scala.collection.mutable.WrappedArray
import org.apache.spark.mllib.linalg.distributed.RowMatrix

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
    val src = spark.textFile(inputpath + "kmeans1")
    val srcDS = src.map {
      line =>
        var data = line.split(",")
        RawDataRecord(data(0).toInt, data(1), data(2), data(3).toInt)
    }

    //70%作为训练数据，30%作为测试数据
    //    val splits = srcDS.randomSplit(Array(0.7, 0.3))
    //将打标签的数据转换成DataFrame数据格式
    val trainingDF = srcDS.toDF()
    //    val testDF = src.toDF()

    //将词语转换成数据,Tokenizer是按着空格切分
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
    featurizedData.select($"hashWithTf").take(50).foreach(println)

    //计算每个词的IDF
    val idf = new IDF().setInputCol("hashWithTf").setOutputCol("features")
    //计算每个词的TF-IDF
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.cache
    rescaledData.show
    rescaledData.rdd.repartition(1).saveAsTextFile(newpath + "text_words_hashWithTf_idfnum")
    //对数据的归一化判断
    val rescaledDatanormalized = rescaledData.select($"index", $"features")
    val normalizedMlibVectorRDD = rescaledDatanormalized.rdd.map {
      case Row(index: Int, features: Vector) =>
        val mlvector = Vectors.dense(features.toArray)
        mlvector
    }
    //normalizedMlibVectorRDD是Mlib RDD[Vector] 其中Vector是稀疏矩阵
    normalizedMlibVectorRDD.repartition(1).saveAsTextFile(newpath + "normalizedMlibVector")

    val CardataMatrix = new RowMatrix(normalizedMlibVectorRDD)
    val CarMatrixSummary = CardataMatrix.computeColumnSummaryStatistics()
    println("data features 矩阵平均值:" + CarMatrixSummary.mean)
    println("data features 矩阵方差:" + CarMatrixSummary.variance)
    //感觉是有离群点 应该进行归一化处理
    //结果写入文本文件
    val Matrixoutput = new PrintWriter(newpath + "Matrixoutput")
    Matrixoutput.println("data features 矩阵平均值:" + CarMatrixSummary.mean + "\n" + "data features 矩阵方差:" + CarMatrixSummary.variance + "\n")
    Matrixoutput.close

    //k-means
    //模型参数
    val Knumber = 5
    val MaxIter = 30
    val seednum = 1L
    val kmeans = new KMeans()
      .setK(Knumber)
      .setSeed(seednum)
      .setMaxIter(MaxIter)
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
    val WSSSE = kmeansModel.computeCost(rescaledData)
    //误差计算
    println("with set sum of squared errors:" + WSSSE)
    //初步寻找最佳K值
    val findNumK = new PrintWriter(newpath + "findNumK")
    val savefindK = new PrintWriter(newpath + "savefindK")
    val ks: Array[Int] = Array(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    ks.foreach { k =>
      val Kmeanns = new KMeans()
        .setK(k)
        .setSeed(1L)
        .setMaxIter(30) //最大迭代次数
      val kmeansModel = Kmeanns.fit(rescaledData)
      val ssd = kmeansModel.computeCost(rescaledData)
      findNumK.println("sum of squared distances of points to their nearest center when k=" + k + " -> " + ssd)
      savefindK.println(ssd) //3-20
    }
    findNumK.close
    savefindK.close

    //保存k-means结果rdd
    val predictioned = kmeansModel.summary.predictions
    predictioned.cache()
    //取前10行的聚类结果指定列输出到控制台观察结果
    kmeansModel.summary.predictions
      .select($"prediction", $"text")
      .take(10)
      .foreach { println }
    predictioned.show(50)

    //保存带有聚类了类簇标签的文件到指定文件夹
    predictioned
      .rdd
      .repartition(1)
      .saveAsTextFile(newpath + "kmeans-summary")

    val datapre = predictioned
      .select($"index", $"words", $"features", $"prediction")
      .rdd

    datapre.repartition(1).saveAsTextFile(newpath + "datapre")
    //方法：计算向量的欧拉距离
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = pow(v1 - v2, 2).sum
    //case匹配需要的属性
    val dataVectorComputeDist = datapre.map {
      case Row(index: Int, words: WrappedArray[String], features: Vector, prediction: Int) =>
        val Center = kmeansModel.clusterCenters(prediction)
        val dist = computeDistance(DenseVector(Center.toArray), DenseVector(features.toArray))
        (index, prediction, dist, words, features)
    }
    dataVectorComputeDist.cache
    //[格式]：index,所属聚类编号,与聚类中心距离,分词向量,tf-idf向量
    //变成一个(k,v)的Map型 k是prediction
    val clusterAssignments = dataVectorComputeDist.groupBy {
      case (index, prediction, dist, words, features) => prediction
    }.collectAsMap
    //输出每个类前10个向量结果 按与其聚类中心距离排序(K,V)按K排序
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      println(s"聚类中心 $k:")
      val m = v.toSeq.sortBy(_._3)
      print(m.take(10).map {
        case (index, prediction, dist, words, features) => (index, prediction, dist, words)
      }.mkString("\n"))
      println("\n")
    }
    //对每个小类进行分组处理
    //    val preAssignments = dataVectorComputeDist.groupBy {
    //      case (index, prediction, dist, words, features) => prediction
    //    }
    //先按着TF-IDF的思路做一下试试
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      val value = v.toSeq.sortBy(_._3)
      val vec = value.take(10).map {
        case (index, prediction, dist, words, features) =>
          (index, words, dist, prediction)
          //重新打标签
          NewDataRecord(index, words, dist, prediction)
      }
      val vecDF = vec.toDF()
      vecDF.show
      val hashingTFIncluster = new HashingTF()
        .setNumFeatures(1000)
        .setInputCol("words")
        .setOutputCol("NewhashWithTf")
      val vecHashDF = hashingTFIncluster.transform(vecDF)
      //新的features 在类内加权
      val newidf = new IDF().setInputCol("NewhashWithTf").setOutputCol("features")
      val NewidfModel = newidf.fit(vecHashDF)
      val vecTFIDF = NewidfModel.transform(vecHashDF)
      val wordsWithFeatureDF = vecTFIDF.select($"words", $"features", $"prediction")
      val wordsWithFeatureDFRdd = wordsWithFeatureDF.rdd.map {
        case Row(words: WrappedArray[String], features: Vector, prediction: Int) =>
          (words, features, prediction)
      }
      wordsWithFeatureDFRdd.repartition(1).saveAsTextFile(newpath + s"wordsWithFeatureDFRdd$k")
      val RDDcount = wordsWithFeatureDFRdd.count
      if (RDDcount > 1) {
        val HotWords = wordsWithFeatureDFRdd.map {
          case (words, features, prediction) =>
            val maxHash = features.toSparse.argmax
            val HashArray = features.toSparse.indices
            val maxIndex = HashArray.indexOf(maxHash)
            (words(maxIndex))
        }
      }
      //      println("聚类中心" + k + "的每一条文档tf-idf最高词汇：")
      //      HotWords.map { x => print(x) }
      //      HotWords.repartition(1).saveAsTextFile(newpath+s"Hotword$k")
//修改1

    }

  }
}

case class RawDataRecord(index: Int, text: String, time: String, recall: Int)
case class NewDataRecord(index: Int, words: WrappedArray[String], dist: Double, prediction: Int)
 



























































