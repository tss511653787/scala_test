package HotalDataHotWords
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
import scala.collection.JavaConversions
import java.util.ArrayList

object LdaFindTopicWord {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("LDA-test")
    val spark = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    val inputpath = "C:/Users/Administrator/Desktop/"
    val outputpath = "C:/Users/Administrator/Desktop/TF-IDF/"
    val src = spark.textFile(inputpath + "tfinput")
    src.cache()
    val srcDS = src.map {
      line =>
        var data = line.split(",")
        TFdatarecord(data(0).toInt, data(1), data(2).toDouble, data(3), data(4))
    }
    srcDS.cache()
    val LDAinputDF = srcDS.toDF()
    LDAinputDF.show()
    LDAinputDF.cache()
    //将text按空格切分转化为数组 WrappedArray
    val rextokenizer = new Tokenizer()
      .setInputCol("splitwords")
      .setOutputCol("words")
    val wordsData = rextokenizer.transform(LDAinputDF)
    wordsData.cache()
    wordsData.show

    val countvectorizer = new CountVectorizer()
    countvectorizer.setInputCol("words").setOutputCol("LDAvec")
    val countvectorizermodel = countvectorizer.fit(wordsData)
    val LDAWithvec = countvectorizermodel.transform(wordsData)
    LDAWithvec.cache()
    val wordarr = countvectorizermodel.vocabulary
    val idf = new IDF()
      .setInputCol("LDAvec")
      .setOutputCol("TfIdfvec")
    val idfmodel = idf.fit(LDAWithvec)
    val newDF = idfmodel.transform(LDAWithvec)
    newDF.cache()
    newDF.show()
    newDF.rdd.repartition(1).saveAsTextFile(outputpath + "newDF")
    //提取关键词
    val TFIDFvec = newDF.rdd.map {
      case Row(index: Int, splitwords: String, score: Double, comments: String, customer_type: String, words: WrappedArray[String], ldavec: Vector, tfIdfvec: Vector) =>
        (index, tfIdfvec)
    }
    TFIDFvec.cache()

    val HotWords = TFIDFvec.map {
      case (index, SparseVector(size, indices, values)) =>
        var hashnumarr = indices
        var tfidfvalues = values
        val arrbuf = new ArrayBuffer[String]
        for (i <- 1 to 4) {
          val findmax = tfidfvalues.indexOf(tfidfvalues.max)
          //找到最大值后置0
          tfidfvalues(findmax) = 0.0
          val maxnum = hashnumarr(findmax)
          // arrbuf += wordarr(maxnum)
          arrbuf += wordarr(maxnum)
        }
        val arr = arrbuf.mkString("", " ", "")
        (index, arr)
    }
    HotWords.cache()
    val DFhotwords = HotWords.toDF()
    DFhotwords.cache()
    val DFhot = DFhotwords.select(DFhotwords("_1").as("index"), DFhotwords("_2").as("docHotwords"))
    DFhot.cache()
    val joinTodata = DFhot.join(newDF, "index")
    joinTodata.cache()
    val outDF = joinTodata.select("index", "score", "comments", "customer_type", "splitwords", "docHotwords")
    outDF.cache()
    outDF.show()
    outDF.rdd.repartition(1).saveAsTextFile(outputpath + "output")

    //对整个数据集合进行热门词语提取
    //1直接使用splitword加权观察结果
    val textrank1Input = joinTodata.select("words").rdd.map {
      case Row(words: WrappedArray[String]) =>
        words.toList
    }
    textrank1Input.cache()
    //对整个聚类提取进行rank
    val arm = new ArrayBuffer[String]
    val collwords = textrank1Input.collect
    for (i <- 0 to collwords.length - 1) {
      arm ++= collwords(i)
    }
    val armlist = arm.toList
    val clusterRank = new PrintWriter(outputpath + "clusterRank")
    val Keyword = KeywordExtractor.keywordExtractor("Url", 7, armlist, 50, 100, 0.85f)
    clusterRank.println("热词")
    Keyword.foreach(clusterRank.println)
    clusterRank.close()

    //2使用每条文档关键词观察结果
    val textrank2Input = joinTodata.select("docHotwords").rdd.map {
      case Row(docHotwords: String) =>
        val toarr = docHotwords.split(" ")
        toarr.toList
    }
    val arm2 = new ArrayBuffer[String]
    val collwords2 = textrank2Input.collect
    for (i <- 0 to collwords2.length - 1) {
      arm2 ++= collwords2(i)
    }
    val armlist2 = arm2.toList
    val clusterRank2 = new PrintWriter(outputpath + "clusterRank2")
    val Keyword2 = KeywordExtractor.keywordExtractor("Url", 7, armlist2, 50, 100, 0.85f)
    clusterRank2.println("热词")
    //Keyword2.foreach(x => clusterRank2.print(x._1 + " "))
    Keyword2.foreach(clusterRank2.println)

    clusterRank2.close()

    //LDA提取topic
    val lda = new LDA()
      .setK(30)
      .setMaxIter(100)
      .setOptimizer("em")
      .setFeaturesCol("LDAvec")
    val ldamodel = lda.fit(joinTodata)
    //设置每个topic的前n个代表词
    val wordnum = 10
    val topic = ldamodel
      .describeTopics(wordnum)
    topic.cache()
    val hot = topic.rdd.map {
      case Row(topic: Int, termIndices: WrappedArray[Int], termWeights: WrappedArray[Long]) =>
        val str = ArrayBuffer[String]()
        for (i <- 0 to wordnum - 1) {
          str += wordarr(termIndices(i))
        }
        ("Topic" + topic, str.mkString("", " ", ""))
    }
    hot.cache()
    //整理转换
    val hottopicwordDF = hot.toDF()
    hottopicwordDF.cache()
    val rename = hottopicwordDF.select(hottopicwordDF("_1").as("topic"), hottopicwordDF("_2").as("words"))
    rename.cache()
    val topicWithword = rename.toDF()
    topicWithword.show
    topicWithword.rdd.repartition(1).saveAsTextFile(outputpath + "TopicHotWord")
    //  spark.stop()
    
    //提取前后词语
    val findIn = newDF.rdd.map {
      case Row(index: Int, splitwords: String, score: Double, comments: String, customer_type: String, words: WrappedArray[String], ldavec: Vector, tfIdfvec: Vector) =>
        (words.toArray)
    }
    val findbuf = new ArrayBuffer[String]
    val findInArr = findIn.collect
    for (i <- 0 to findInArr.length - 1) {
      findbuf ++= findInArr(i)
    }
    val findarr = findbuf.toArray
    val wordsArr = Array("服务", "设施", "早餐", "位置", "环境", "性价比",
      "装修", "价格", "房费", "卫生", "房间", "大堂", "娱乐", "健身", "停车场",
      "入住", "效率", "服务态度", "交通", "卫生间", "安静", "干净", "咖啡厅", "网络",
      "周边", "公交", "地铁", "前台", "空调", "热水")
    for (word <- wordsArr) {
      findwords(findarr, word, outputpath)
    }

  }
  def findwords(strArr: Array[String], word: String, outpath: String) {
    val answer = new ArrayList[String]()
    for (i <- 0 to strArr.length - 1) {
      var temp = new StringBuilder()
      if (strArr(i).equals(word)) {
        temp ++= "(" + strArr(i) + ")"
        //找到词
        var k = i
        //将前面4个人词加入到temps
        var m = 0
        while (((k - 1) != -1) && (m != 3)) {
          temp.insert(0, strArr(k - 1) + " ")
          m = m + 1
          k = k - 1
        }
        //将后面4个加入到temp
        temp.append(" ")
        k = i
        var n = 0
        while (((k + 1) != strArr.length) && (n != 3)) {
          temp.append(strArr(k + 1) + " ")
          n = n + 1
          k = k + 1
        }
        //加入到answer
        answer.add(temp.toString())
      }
    }
    //输出到文件
    val out = new PrintWriter(outpath + word)
    val res = answer.toArray()
    res.foreach(out.println)
    out.close()
  }
}
case class TFdatarecord(index: Int, splitwords: String, score: Double, comments: String, customer_type: String)