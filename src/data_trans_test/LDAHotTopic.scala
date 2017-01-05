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
import org.apache.spark.ml.feature.IDF
import java.io.PrintWriter
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.linalg.Vector
import scala.collection.mutable.WrappedArray
import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.feature.Word2Vec
import testrank.KeywordExtractor

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
    val src = spark.textFile(inputpath + "kmeans_noST_noLC")
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
    val topicnum = 10
    val maxiter = 100
    val Optimizermethods = "em"
    //EM消耗大量内存 Online更快
    val LDAinput = LDAWithvec.select("index", "words", "LDAvec", "time", "recall")
    val lda = new LDA()
      .setK(topicnum)
      .setMaxIter(maxiter)
      .setOptimizer(Optimizermethods)
      .setFeaturesCol("LDAvec")
    val ldamodel = lda.fit(LDAinput)
    //模型描述
    //println("当前模型参数描述:")
    //println(ldamodel.explainParams())
    //列举每个主题中权重最大的10个词语
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
    //词语-主题矩阵 列代表每个词语在每个主题上的概率分布 行书是每个不重复词语
    val topicsMat = ldamodel.topicsMatrix
    println("文档-主题矩阵：")
    println(topicsMat.toString)

    //整理转换
    val hottopicwordDF = hot.toDF()
    val rename = hottopicwordDF.select(hottopicwordDF("_1").as("topic"), hottopicwordDF("_2").as("words"))
    val topicWithword = rename.toDF()
    topicWithword.show
    hot.repartition(1).saveAsTextFile(outputpath + "hot")

    //模型评估
    //模型的评价指标：logLikelihood，logPerplexity
    //（1）根据训练集的模型分布计算的log likelihood(对数似然率)，越大越好

    //    val ll = ldamodel.logLikelihood(LDAinput)
    //    println("主题数" + topicnum + "的对数似然率:" + ll)
    //（2）Perplexity(复杂度)评估，越小越好
    //    val lp = ldamodel.logPerplexity(LDAinput)
    //    println("主题数" + topicnum + "的复杂度上界:" + lp)

    //对参数进行调试
    //对迭代次数进行调试
    //    val llouput = new PrintWriter(outputpath + "llouput")
    //    val lpouput = new PrintWriter(outputpath + "lpouput")
    //    for (i <- Array(5, 10, 20, 40, 60, 120, 200, 500)) {
    //      val testlda = new LDA()
    //        .setK(topicnum)
    //        .setMaxIter(i)
    //        .setOptimizer("online")
    //        .setFeaturesCol("LDAvec")
    //      val testmodel = testlda.fit(LDAinput)
    //      val testll = testmodel.logLikelihood(LDAinput)
    //      val testlp = testmodel.logPerplexity(LDAinput)
    //      llouput.print(testll + "\n")
    //      lpouput.print(testlp + "\n")
    //    }
    //    llouput.close
    //    lpouput.close
    //主题数目K对logLikelihood值的影响
    //问题：可能由于目前数据量很小 k值在3-20间logll值一直递减
    //    val numKlogll = new PrintWriter(outputpath + "numKlogll")
    //    for (i <- Array(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)) {
    //      val testlda = new LDA()
    //        .setK(i)
    //        .setMaxIter(100)
    //        .setOptimizer("online")
    //        .setFeaturesCol("LDAvec")
    //      val testmodel = testlda.fit(LDAinput)
    //      val testll = testmodel.logLikelihood(LDAinput)
    //      numKlogll.print(testll + "\n")
    //    }
    //    numKlogll.close
    //EM 方法，分析DocConcentration的影响，算法默认值是(50/k)+1
    //    val DocConcentrationloglp = new PrintWriter(outputpath + "DocConcentration")
    //    for (i <- Array(1.2, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)) {
    //      val lda = new LDA()
    //        .setK(5)
    //        .setTopicConcentration(1.1)
    //        .setDocConcentration(i)
    //        .setOptimizer("online")
    //        .setMaxIter(30)
    //        .setFeaturesCol("LDAvec")
    //      val model = lda.fit(LDAinput)
    //      val lp = model.logPerplexity(LDAinput)
    //      DocConcentrationloglp.print(lp + "\n")
    //    }
    //    DocConcentrationloglp.close

    //对语料进行聚类
    val topicProb = ldamodel.transform(LDAinput)
    topicProb.show
    topicProb.rdd.repartition(1).saveAsTextFile(outputpath + "topicProb")
    //打类别标签
    val topicdis = topicProb.map {
      case Row(index: Int, words: WrappedArray[String], ldavec: Vector, time: String, recall: Int, topicDistribution: Vector) =>
        val arrDouble = topicDistribution.toArray
        val maxindex = arrDouble.indexOf(arrDouble.max)
        val maxprobability = arrDouble.max
        (index, words, ldavec, topicDistribution, maxprobability, maxindex, time, recall)
    }
    val rawdatapre = topicdis.toDF()
    //对列重命名
    val datapre = rawdatapre.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "words")
      .withColumnRenamed("_3", "LDAvec")
      .withColumnRenamed("_4", "topicDistribution")
      .withColumnRenamed("_5", "maxprobability")
      .withColumnRenamed("_6", "prediction")
      .withColumnRenamed("_7", "time")
      .withColumnRenamed("_8", "recall")
    datapre.show
    datapre.cache
    val assignments = datapre.rdd.map {
      case Row(index: Int, words: WrappedArray[String], ldavec: Vector, topicDistribution: Vector, maxprobability: Double, prediction: Int, time: String, recall: Int) =>
        (index, words, ldavec, topicDistribution, maxprobability, prediction, time, recall)
    }
    assignments.cache()
    val clusterAssignments = assignments.groupBy {
      case (index, words, ldavec, topicDistribution, maxprobability, prediction, time, recall) => prediction
    }.collectAsMap
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      //按着文档对出题的概率进行降序排列
      val value = v.toSeq.sortWith(_._5 > _._5)
      //提取概率最高的前10个文档做为输入
      //取得和聚类中心最近文档的数量takenum
      println(s"聚类$k" + "的文档条数：" + value.length)
      val takenum = 15
      val vec = value.take(takenum)
      val vecDF = vec.toDF()
      val allvecDF = value.toDF()
      //聚类中所有文档的df
      val reallvecDF = allvecDF.withColumnRenamed("_1", "index")
        .withColumnRenamed("_2", "words")
        .withColumnRenamed("_3", "LDAvec")
        .withColumnRenamed("_4", "topicDistribution")
        .withColumnRenamed("_5", "maxprobability")
        .withColumnRenamed("_6", "prediction")
        .withColumnRenamed("_7", "time")
        .withColumnRenamed("_8", "recall")
      val numSum = reallvecDF.count.toInt
      //聚类中里中心最近前10文档的
      //使用LDA聚类后向量格式:
      // vec([索引][词语][词频向量][该文档主题概率分布][最大概率][聚族号])
      val renamevecDF = vecDF.withColumnRenamed("_1", "index")
        .withColumnRenamed("_2", "words")
        .withColumnRenamed("_3", "LDAvec")
        .withColumnRenamed("_4", "topicDistribution")
        .withColumnRenamed("_5", "maxprobability")
        .withColumnRenamed("_6", "prediction")
        .withColumnRenamed("_7", "time")
        .withColumnRenamed("_8", "recall")
      //      renamevecDF.rdd.repartition(1).saveAsTextFile(outputpath + s"vecDF$k")
      val clusterRdd = renamevecDF.rdd
      //调用java类创建目录
      SaveFile.makeDir(outputpath + "clustervec/")
      val saveclusterRdd = new PrintWriter(outputpath + "clustervec/" + s"clustervec$k")
      val clusterRddtoArr = clusterRdd.collect
      clusterRddtoArr.foreach {
        case Row(index: Int, words: WrappedArray[String], ldavec: Vector, topicDistribution: Vector, maxprobability: Double, prediction: Int, time: String, recall: Int) =>
          saveclusterRdd.print(numSum + " ")
          saveclusterRdd.print(index + " ")
          saveclusterRdd.print(topicDistribution.toString.replace("[", "").replace("]", "") + " ")
          saveclusterRdd.print(maxprobability + " ")
          saveclusterRdd.print(prediction + " ")
          val newtime = time.split(" ")
          saveclusterRdd.print(newtime(0) + "," + newtime(1) + " ")
          saveclusterRdd.print(recall)
          saveclusterRdd.println
      }
      saveclusterRdd.close

      //构建TF-Idf向量
      val idf = new IDF()
        .setInputCol("LDAvec")
        .setOutputCol("TfIdfvec")
      val idfmodel = idf.fit(renamevecDF)
      val newDF = idfmodel.transform(renamevecDF)
      SaveFile.makeDir(outputpath + "newDF/")
      newDF.rdd.repartition(1).saveAsTextFile(outputpath + "newDF/" + s"newDF$k")
      newDF.show
      //从概率最高的10条文档中找到关键词 TF-IDF最大的10个词语
      val TFIDFvec = newDF.rdd.map {
        case Row(index: Int, words: WrappedArray[String], ldavec: Vector, topicDistribution: Vector, maxprobability: Double, prediction: Int, time: String, recall: Int, tfIdfvec: Vector) =>
          tfIdfvec
      }
      val HotWords = TFIDFvec.map {
        case SparseVector(size, indices, values) =>
          var hashnumarr = indices
          var tfidfvalues = values
          val arrbuf = new ArrayBuffer[String]
          for (i <- 1 to 3) {
            //寻找每个向量中tf-idf最大的3个词语
            val findmax = tfidfvalues.indexOf(tfidfvalues.max)
            //找到最大值后置0
            tfidfvalues(findmax) = 0.0
            val maxnum = hashnumarr(findmax)
            arrbuf += wordarr(maxnum)
          }
          val arr = arrbuf.toArray
          arr
      }

      //TextRank
      //textrank 算法对每条文档进行关键词提取
      val rankvec = renamevecDF.rdd.map {
        case Row(index: Int, words: WrappedArray[String], ldavec: Vector, topicDistribution: Vector, maxprobability: Double, prediction: Int, time: String, recall: Int) =>
          words.toList
      }
      //对整个聚类提取进行rank
      val arm = new ArrayBuffer[String]
      val collwords = rankvec.collect
      for (i <- 0 to collwords.length - 1) {
        arm ++= collwords(i)
      }
      val armlist = arm.toList
      SaveFile.makeDir(outputpath + "clusterRank/")
      val clusterRank = new PrintWriter(outputpath + "clusterRank/" + s"clusterRank$k")
      val Keyword = KeywordExtractor.keywordExtractor("Url", 5, armlist, 10, 100, 0.85f)
      clusterRank.println(s"聚类:$k" + "热词")
      Keyword.foreach(clusterRank.println)
      clusterRank.close()

      //保存第K个聚类结果每条文档热词
      HotWords.cache()
      SaveFile.makeDir(outputpath + "Hotwords/")
      val Savehot = new PrintWriter(outputpath + "Hotwords/" + s"Hotwords$k")
      val Collhot = HotWords.collect()
      var p = 1
      Collhot.foreach { line =>
        Savehot.print(p + " ")
        for (str <- line) Savehot.print(str + " ")
        Savehot.println
        p = p + 1
      }
      Savehot.close

      //以后完善 可以利用wordVec2 模型对提取到的关键词进行语意扩展
      //WordVec2
      //利用wordVec2 模型对提取到的关键词进行语意扩展
      //wordvec2arrbu存贮对单个单词的语意扩展词组
      val wordvec2word = reallvecDF.rdd.map {
        case Row(index: Int, words: WrappedArray[String], ldavec: Vector, topicDistribution: Vector, maxprobability: Double, prediction: Int, time: String, recall: Int) =>
          words.toSeq
      }
      //设置最小词频
      //目前需要所有词所以先设置为1
      //词库是一个聚类中的所有词语
      val minfrewords = 1
      val wordvec = new Word2Vec()
      wordvec.setMinCount(minfrewords)
      val wordvec2model = wordvec.fit(wordvec2word)
      //热词和扩展次的RDD
      val hotWithsim = HotWords.map {
        array =>
          var saveall = new ArrayBuffer[String]
          var num = 0
          for (words <- array) {
            var temparr = wordvec2model.findSynonyms(words, 2)
            saveall += words
            //暂时不实用word2vec进行近义词搜索
            //            saveall += temparr(0)._1
            //            saveall += temparr(1)._1
          }
          saveall.toArray
      }
      val allhotwords = hotWithsim.collect
      val temp = new ArrayBuffer[String]
      for (i <- 0 to allhotwords.length - 1) {
        temp ++= allhotwords(i)
      }
      val tem = temp.toList
      //保存热词结果
      SaveFile.makeDir(outputpath + "clusterhotword/")
      val clusterhotwords = new PrintWriter(outputpath + "clusterhotword/" + s"clusterhotword$k")
      val keyword = KeywordExtractor.keywordExtractor("url", 5, tem, 10, 100, 0.85f)
      clusterhotwords.println(s"聚类:$k" + "热词")
      keyword.foreach(clusterhotwords.println)
      clusterhotwords.close()

      //      val rankHot = hotWithsim.map { list =>
      //        //url:图名称
      //        //5滑动窗口大小
      //        //10关键词个数
      //        //100迭代次数
      //        //0.85阻尼系数
      //        val keyword = KeywordExtractor.keywordExtractor("url", 5, list, 3, 100, 0.85f)
      //        (keyword(0)._1, keyword(1)._1, keyword(2)._1)
      //      }

      //保存第K个聚类结果每条文档热词+扩展词语结果
      //格式：
      //No words1:sim1,sim2,sim3 words2:sim1,sim2,sim3
      HotWords.cache()
      SaveFile.makeDir(outputpath + "HotwordsWithSim/")
      val savehot = new PrintWriter(outputpath + "HotwordsWithSim/" + s"HotwordsWithSim$k")
      val collhot = HotWords.collect()
      var i = 1
      collhot.foreach { line =>
        val linenum = collhot.length
        savehot.print(i + " ")
        for (str <- line) {
          savehot.print(str + ":")
          //利用word2vec模型找3个近义词
          var temparr = wordvec2model.findSynonyms(str, 2)
          for (k <- 0 to 1) {
            savehot.print(temparr(k)._1)
            if (k != 2) savehot.print(",")
          }
          savehot.print(" ")
        }
        savehot.println
        i = i + 1
      }
      savehot.close

    }

  }
}