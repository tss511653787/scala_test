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
import org.apache.spark.ml.clustering.LDA
import scala.collection.mutable.WrappedArray
import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter

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
    //语料词库数组 下标对应
    val wordarr = countvectorizermodel.vocabulary
    LDAWithvec.show
    LDAWithvec.cache()
    //LDA算法
    //LDA模型训练
    val topicnum = 5
    val maxiter = 30
    val LDAinput = LDAWithvec.select("index", "LDAvec")
    val lda = new LDA()
      .setK(topicnum)
      .setMaxIter(maxiter)
      .setOptimizer("em")
      .setFeaturesCol("LDAvec")
    val ldamodel = lda.fit(LDAinput)
    //模型描述
    val wordnum = 10
    val topic = ldamodel
      .describeTopics(wordnum)
    //主题词语矩阵 每个主题中权重最大的n个词语
    topic.show()
    val hotword = topic.rdd.map {
      case Row(topic: Int, termIndices: WrappedArray[Int], termWeights: WrappedArray[Long]) =>
        (topic, termIndices)
    }
    val hot = hotword.map {
      case (topic: Int, termIndices: WrappedArray[Int]) =>
        val str = ArrayBuffer[String]()
        for (i <- 0 to wordnum - 1) {
          str += wordarr(termIndices(i))
        }
        (topic, str)
    }
    //整理转换
    val hottopicwordDF = hot.toDF()
    val rename = hottopicwordDF.select(hottopicwordDF("_1").as("topic"), hottopicwordDF("_2").as("words"))
    val topicWithword = rename.toDF()
    topicWithword.show
    hot.repartition(1).saveAsTextFile(outputpath + "hot")

    //模型评估
    //模型的评价指标：logLikelihood，logPerplexity
    //（1）根据训练集的模型分布计算的log likelihood(对数似然率)，越大越好
    val ll = ldamodel.logLikelihood(LDAinput)
    println("主题数" + topicnum + "的对数似然率:" + ll)
    //（2）Perplexity(复杂度)评估，越小越好
    val lp = ldamodel.logPerplexity(LDAinput)
    println("主题数" + topicnum + "的复杂度上界:" + lp)

    //对语料进行聚类
    val topicProb = ldamodel.transform(LDAinput)
    topicProb.show
    topicProb.rdd.repartition(1).saveAsTextFile(outputpath + "topicProb")
    //对参数进行调试
    //对迭代次数进行调试
    val llouput = new PrintWriter(outputpath + "llouput")
    val lpouput = new PrintWriter(outputpath + "lpouput")
    for (i <- Array(5, 10, 20, 40, 60, 120, 200, 500)) {
      val testlda = new LDA()
        .setK(topicnum)
        .setMaxIter(i)
        .setOptimizer("online")
        .setFeaturesCol("LDAvec")
      val testmodel = testlda.fit(LDAinput)
      val testll = testmodel.logLikelihood(LDAinput)
      val testlp = testmodel.logPerplexity(LDAinput)
      llouput.print(testll + "\n")
      lpouput.print(testlp + "\n")
    }
    llouput.close
    lpouput.close
    //主题数目K对logLikelihood值的影响
    val numKlogll = new PrintWriter(outputpath + "numKlogll")
    for (i <- Array(3, 4, 5, 6, 7, 8, 9, 10, 11, 12)) {
      val testlda = new LDA()
        .setK(i)
        .setMaxIter(30)
        .setOptimizer("online")
        .setFeaturesCol("LDAvec")
      val testmodel = testlda.fit(LDAinput)
      val testll = testmodel.logLikelihood(LDAinput)
      numKlogll.print(testll + "\n")
    }
    numKlogll.close
    //EM 方法，分析setDocConcentration的影响，计算(50/k)+1=50/5+1=11
    val DocConcentrationloglp = new PrintWriter(outputpath + "DocConcentration")
    for (i <- Array(1.2, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)) {
      val lda = new LDA()
        .setK(5)
        .setTopicConcentration(1.1)
        .setDocConcentration(i)
        .setOptimizer("em")
        .setMaxIter(30)
        .setFeaturesCol("LDAvec")
      val model = lda.fit(LDAinput)
      val lp = model.logPerplexity(LDAinput)
      DocConcentrationloglp.print(lp + "\n")
    }
    DocConcentrationloglp.close

    //词语-主题矩阵 列代表每个词语在每个主题上的概率分布 行书是每个不重复词语
    val topicsMat = ldamodel.topicsMatrix
    println("词语-主题矩阵：")
    println(topicsMat.toString)

  }
}