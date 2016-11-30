import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.log4j.Level
import org.apache.log4j.Logger

/**
 * Created by TSS on 16/11/30.
 */
object KeywordExtractorTest {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("KeywordExtractorTest").setMaster("local")
    val sc = new SparkContext(conf)

    val file = sc.textFile("C:/Users/dell/Desktop/data/data_text_car.txt")

    val docs = file.map { row =>

      row.split(" ")
    }

    val keyWordList = docs.map(doc => KeywordExtractor.keywordExtractor("url", 5, doc.toList, 10, 100, 0.85f))

    var i = 1

    keyWordList.foreach { doc =>
      {
        println(s"第${i}篇文章的关键词")

        doc.foreach(x => print(s"${x._1}" + x._2 + "\t"))

        println()
        i = i + 1
      }
    }

    sc.stop()

  }

}
