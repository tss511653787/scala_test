import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{ SparseVector => SV }
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import breeze.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
object Newsgroups {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[4]", "20Newsgroups")
    val path = "hdfs://tss.hadoop2-1:8020/user/root/dataSet/20news-bydate/20news-bydate-train/*"
    //val path = "20news-bydate/20news-bydate-train/*"
    val rdd = sc.wholeTextFiles(path)
    //println(rdd.count)
    val newsgroups = rdd.map { case (file, text) => file.split("/").takeRight(2).head }
    val countByGroup = newsgroups.map(n => (n, 1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
    println(countByGroup)
    //分词
    val text = rdd.map { case (file, text) => text }
    val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase)) //英文只是简单的按着空格分词
    println(whiteSpaceSplit.distinct.count)
    val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))

    val regex = """[^0-9]*""".r
    val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
    val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
    val oreringDesc = Ordering.by[(String, Int), Int](_._2)
    val stopwords = Set(
      "the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not", "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to")
    val tokenCountsFilteredStopwords = tokenCounts.filter { case (k, v) => !stopwords.contains(k) }
    val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter { case (k, v) => k.size >= 2 }
    val oreringAsc = Ordering.by[(String, Int), Int](-_._2)
    val rareTokens = tokenCounts.filter { case (k, v) => v < 2 }.map { case (k, v) => k }.collect.toSet
    val tokenCountlteredAll = tokenCountsFilteredSize.filter { case (k, v) => !rareTokens.contains(k) }
    //总结创建分词函数
    def tokenize(line: String): Seq[String] = {
      line.split("""\W+""")
        .map(_.toLowerCase)
        .filter(token => regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token))
        .filterNot(token => rareTokens.contains(token))
        .filter(token => token.size >= 2)
        .toSeq
    }
    //训练TF*IDF模型
    val tokens = text.map(doc => tokenize(doc))
    val dim = math.pow(2, 18).toInt
    val hashingTF = new HashingTF(dim)

    val tf = hashingTF.transform(tokens)
    tf.cache
    val v = tf.first.asInstanceOf[SV]
    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)
    val v2 = tfidf.first.asInstanceOf[SV]
    val minMaxVals = tfidf.map { v =>
      val sv = v.asInstanceOf[SV]
      (sv.values.min, sv.values.max)
    }
    val globalMinMax = minMaxVals.reduce {
      case ((min1, max1), (min2, max2)) =>
        (math.min(min1, min2), math.max(max1, max2))
    }
    val common = sc.parallelize(Seq(Seq("you", "do", "we")))
    val tfCommon = hashingTF.transform(common)
    val tfidfCommon = idf.transform(tfCommon)
    val commonVector = tfidfCommon.first.asInstanceOf[SV]
    println(commonVector.values.toSeq)
    val uncommon = sc.parallelize(Seq(Seq("telescope", "legislation", "investment")))
    val tfUncommon = hashingTF.transform(uncommon)
    val tfidfUncommon = idf.transform(tfUncommon)
    val uncommonVector = tfidfUncommon.first.asInstanceOf[SV]
    println(uncommonVector.values.toSeq)
    //训练文档相似度
    val hockeyText = rdd.filter { case (file, text) => file.contains("hockey") }
    val hockeyTF = hockeyText.mapValues(doc => hashingTF.transform(tokenize(doc)))
    val hockeyTfIdf = idf.transform(hockeyTF.map(_._2))
    val hockey1 = hockeyTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breeze1 = new SparseVector(hockey1.indices, hockey1.values, hockey1.size)
    val hockey2 = hockeyTfIdf.sample(true, 0.1, 43).first.asInstanceOf[SV]
    val breeze2 = new SparseVector(hockey2.indices, hockey2.values, hockey2.size)
    val cosineSim = breeze1.dot(breeze2) / (norm(breeze1) * norm(breeze2))
    println(cosineSim)
    //
    val newsgroupsMap = newsgroups.distinct.collect().zipWithIndex.toMap
    val zipped = newsgroups.zip(tfidf)
    val train = zipped.map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector) }
    train.cache
    val model = NaiveBayes.train(train, lambda = 0.1)
    //
    val testPath = "20news-bydate/20news-bydate-test/*"
    val testRDD = sc.wholeTextFiles(testPath)
    val testLabels = testRDD.map {
      case (file, text) =>
        val topic = file.split("/").takeRight(2).head
         newsgroupsMap(topic)
    }
    val testTf = testRDD.map { case (file, text) => hashingTF.transform(tokenize(text)) }
    val testTfIdf = idf.transform(testTf)
    val zippedTest = testLabels.zip(testTfIdf)
    val test = zippedTest.map { case (topic, vector) => LabeledPoint(topic, vector) }
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println(accuracy)
    val metrics = new MulticlassMetrics(predictionAndLabel)
    println(metrics.weightedFMeasure)

  }
}
