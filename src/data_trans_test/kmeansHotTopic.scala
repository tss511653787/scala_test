package data_trans_test

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.linalg.{ SparseVector => mlSparseVector }
import org.apache.spark.mllib.feature.{ HashingTF => mlibHashingTF }
import org.apache.spark.mllib.feature.{ IDF => mlibIDF }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{ Vector => mllibVector }
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.SparseVector
import scala.reflect.runtime.universe
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.io.PrintWriter
import breeze.linalg._
import breeze.numerics.pow
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.WrappedArray
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.mllib.feature.{ Word2Vec => mlibWord2Vec }

object kmeansFindPaperHotTopic {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
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
    val src = spark.textFile(inputpath + "kmeans_cn")
    val srcDS = src.map {
      line =>
        var data = line.split(",")
        RawDataRecord(data(0).toInt, data(1), data(2), data(3).toInt)
    }
    //将打标签的数据转换成DataFrame数据格式
    val trainingDF = srcDS.toDF()
    //    val testDF = src.toDF()
    trainingDF.show
    //将词语转换成数据,Tokenizer是按着空格切分
    val rextokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val wordsData = rextokenizer.transform(trainingDF)
    println("text+words:")
    //建立词库
    val dicword = new CountVectorizer()
    dicword.setInputCol("words").setOutputCol("wordsfrequency")
    val Countmodel = dicword.fit(wordsData)
    val wordarr = Countmodel.vocabulary
    //观察结果
    wordsData.cache
    wordsData.show
    //存储路径
    val newpath = "C:/Users/dell/Desktop/KmeansResult/"

    //计算每个词在文档中的词频
    val hashingTF = new HashingTF()
      .setNumFeatures(10000)
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
    val Knumber = 13
    val MaxIter = 100
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
    //当前k-means++模型参数描述
    println("当前Kmeans++模型的参数描述:")
    println(kmeansModel.explainParams())

    //保存带有聚类了类簇标签的文件到指定文件夹
    predictioned
      .rdd
      .repartition(1)
      .saveAsTextFile(newpath + "kmeans-summary")

    val datapre = predictioned
      .select($"index", $"words", $"features", $"prediction", $"text")
      .rdd

    datapre.repartition(1).saveAsTextFile(newpath + "datapre")
    //方法：计算向量的欧拉距离
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = pow(v1 - v2, 2).sum
    //case匹配需要的属性
    val dataVectorComputeDist = datapre.map {
      case Row(index: Int, words: WrappedArray[String], features: Vector, prediction: Int, text: String) =>
        val Center = kmeansModel.clusterCenters(prediction)
        val dist = computeDistance(DenseVector(Center.toArray), DenseVector(features.toArray))
        (index, prediction, dist, words, features, text)
    }
    dataVectorComputeDist.cache
    //[格式]：index,所属聚类编号,与聚类中心距离,分词向量,tf-idf向量
    //变成一个(k,v)的Map型 k是prediction
    val clusterAssignments = dataVectorComputeDist.groupBy {
      case (index, prediction, dist, words, features, text) => prediction
    }.collectAsMap
    //输出每个类前10个向量结果 按与其聚类中心距离排序(K,V)按K排序
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      println(s"聚类中心 $k:")
      val m = v.toSeq.sortBy(_._3)
      print(m.take(10).map {
        case (index, prediction, dist, words, features, text) => (index, prediction, dist, words, text)
      }.mkString("\n"))
      println("\n")
    }
    //使用mllib包中的TF-IDF方法对每个聚类族中进行热门词汇提取
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      val value = v.toSeq.sortBy(_._3)
      val vec = value.take(10).map {
        case (index, prediction, dist, words, features, text) =>
          (index, words, dist, prediction, text)
          //重新打标签
          NewDataRecord(index, words, dist, prediction, text)
      }
      //将index words dist pre text转化成一个新的DataFrame
      val vecDF = vec.toDF()
      //输出这个新的DF
      vecDF.show
      //提取text属性为后面建立词库
      val rddtext = vecDF.rdd.map {
        case Row(index: Int, words: WrappedArray[String], dist: Double, prediction: Int, text: String) =>
          (text)
      }
      val rddtextseq = rddtext.map {
        line =>
          val words = line.split(" ")
          words.toSeq
      }
      //提取索引属性
      val rddvecIndex = vecDF.rdd.map {
        case Row(index: Int, words: WrappedArray[String], dist: Double, prediction: Int, text: String) =>
          (index)
      }
      //提取index, words, dist, prediction, text属性
      val rddvecdata = vecDF.rdd.map {
        case Row(index: Int, words: WrappedArray[String], dist: Double, prediction: Int, text: String) =>
          (index, words, dist, prediction, text)
      }
      //Hashing必须使用的默认的维度向量 否组可能出错
      //使用mllib中的HashingTF算法加载native哈希方法
      val hashingTFIncluster = new mlibHashingTF()
        .setHashAlgorithm("native")
      val vecHashDF = hashingTFIncluster.transform(rddtextseq)
      val newidf = new mlibIDF()
      val NewidfModel = newidf.fit(vecHashDF)
      val vecTFIDF = NewidfModel.transform(vecHashDF)
      //将训练好的Vector向量和加入到原有的DF中
      //对index和Vector进行对应绑定
      val rddvecIndexWithvecTFIDF = rddvecIndex.zip(vecTFIDF)
      //绑定后传化为DataFrame格式
      val rddvecIndexWithvecTFIDFDF = rddvecIndexWithvecTFIDF.toDF()
      //用select修改列名_1==>index _2==> features
      val rename = rddvecIndexWithvecTFIDFDF.select(rddvecIndexWithvecTFIDFDF("_1").as("index"), rddvecIndexWithvecTFIDFDF("_2").as("features"))
      rename.cache
      //两个DF进行join操作
      val vecjoinIDF = vecDF.join(rename, vecDF("index") === rename("index"))
      //输出连接结果
      vecjoinIDF.show

      //提取每个向量TF-IDF最高的词语
      val wordsWithFeatureDF = vecjoinIDF.select($"words", $"prediction", $"features", $"text")
      val wordsWithFeatureDFRdd = wordsWithFeatureDF.rdd.map {
        case Row(words: WrappedArray[String], prediction: Int, features: mllibVector, text: String) =>
          (words, prediction, features)
      }
      //建立去重词库代码
      val wordsarr = rddtextseq.map { line =>
        val lineword = line.toArray
        lineword
      }
      val arrWords = wordsarr.collect
      //统计该类族中文档条数
      val RDDcount = wordsWithFeatureDFRdd.count.toInt
      //只对聚类中超过1条的类族进行提取(1条进行TF-IDF加权没有意义)
      if (RDDcount > 1) {
        val temp = new ArrayBuffer[String]
        for (i <- 0 to arrWords.length - 1) {
          temp ++= arrWords(i)
        }
        val tem = temp.toArray
        val distinctWords = tem.distinct
        //去重词语保存Hash值
        val savedistinct = new PrintWriter(newpath + "savedistinct")
        distinctWords.foreach {
          words => savedistinct.print(words.hashCode() + " ")
        }
        savedistinct.close
        //提取vector属性
        val HotWordsvec = wordsWithFeatureDFRdd.map {
          case (words, prediction, features) => features
        }
        HotWordsvec.repartition(1).saveAsTextFile(newpath + s"HotWordsvec$k")
        val wordvec2row = wordsWithFeatureDFRdd.toDF()
        val wordvec2re = wordvec2row
          .withColumnRenamed("_1", "words")
          .withColumnRenamed("_2", "prediction")
          .withColumnRenamed("_3", "features")
        val wordvec2input = wordvec2re.rdd.map {
          case Row(words: WrappedArray[String], prediction: Int, features: mllibVector) =>
            words.toSeq
        }
        wordvec2input.repartition(1).saveAsTextFile(newpath + s"wordvecinput$k")
        //设置最小词频
        //目前需要所有词所以先设置为1
        val minfrewords = 1
        val wordvec = new mlibWord2Vec()
        wordvec.setMinCount(minfrewords)
        //训练过程
        val wordvec2model = wordvec.fit(wordvec2input)
        //利用wordvec2模型也可以对文本数据进行向量化
        //用来将词表示为数值型向量的工具，其基本思想是将文本中的词映射成一个 K 维数值向量
        //提取词语代码
        val HotWords = HotWordsvec.map {
          case SparseVector(size, indices, values) =>
            var hashnumarr = indices
            var tfidfvalues = values
            val arrbuf = new ArrayBuffer[String]
            for (i <- 1 to 3) {
              //寻找每个向量中tf-idf最大的3个词语
              val findmax = tfidfvalues.indexOf(tfidfvalues.max)
              //找到最大值后置0
              tfidfvalues(findmax) = 0.0
              val maxhashnum = hashnumarr(findmax)
              for (word <- distinctWords) {
                if (word.hashCode() == maxhashnum) {
                  arrbuf += word
                }
              }
            }
            arrbuf
        }
        //保存第K个聚类结果每条文档热词
        HotWords.repartition(1).saveAsTextFile(newpath + s"HotWordCluster$k")
        //利用wordVec2 模型对提取到的关键词进行语意扩展
        //wordvec2arrbu存贮对单个单词的语意扩展词组
        val similarwords = HotWords.map {
          arrbuf =>
            val simarrbuf = new ArrayBuffer[String]
            for (i <- 0 to arrbuf.length - 1) {
              var temparr = wordvec2model.findSynonyms(arrbuf(i), 3)
              for (i <- 0 to 2) {
                val str = temparr(i)._1
                simarrbuf += str
              }
              simarrbuf += "|"
            }
            simarrbuf
        }
        similarwords.repartition(1).saveAsTextFile(newpath + s"similarwords$k")

      } else {
        //该类簇中只有一条文本
        val countwordsfreRDD = wordsWithFeatureDFRdd.toDF()
        val renameRDD = countwordsfreRDD
          .withColumnRenamed("_1", "words")
          .withColumnRenamed("_2", "prediction")
          .withColumnRenamed("_3", "features")
        val newcountwords = new CountVectorizer()
        newcountwords.setInputCol("words").setOutputCol("wordsFrequency")
        val countwords = newcountwords.fit(renameRDD)
        val rescount = countwords.transform(renameRDD)
        rescount.show
        val wordarr = countwords.vocabulary
        val wordsFreArr = rescount.rdd.map {
          case Row(words: WrappedArray[String], prediction: Int, features: mllibVector, wordsFrequency: Vector) =>
            wordsFrequency
        }
        val Hotwords = wordsFreArr.map {
          case mlSparseVector(size, indices, values) =>
            var hashnumarr = indices
            var tfidfvalues = values
            val arrbuf = new ArrayBuffer[String]
            for (i <- 1 to 3) {
              val findmax = tfidfvalues.indexOf(tfidfvalues.max)
              tfidfvalues(findmax) = 0.0
              val maxhashnum = hashnumarr(findmax)
              arrbuf += wordarr(maxhashnum)
            }
            arrbuf

        }
        Hotwords.repartition(1).saveAsTextFile(newpath + s"HotWordCluster$k")
      }
    }

  }
}

case class RawDataRecord(index: Int, text: String, time: String, recall: Int)
case class NewDataRecord(index: Int, words: WrappedArray[String], dist: Double, prediction: Int, text: String)
case class toCol(vec: mllibVector)
 



























































