import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkConf

object LiPuDataProcess {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("LiPuData")
    val spark = new SparkContext(conf)
    //引入spark sql标签
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    //输入输出路径
    val inPutpath = "C:/Users/dell/Desktop/SogouQ.mini/SogouQ.sample"
    val outPutpath = "C:/Users/dell/Desktop/lPoutput/"
    //读入文件
    val dataInput = spark.textFile(inPutpath)
    //划分属性列
    val data = dataInput.map { line =>
      val splitData = line.split("\t")
      (splitData(0), splitData(1).toLong, splitData(2), splitData(3), splitData(4))
    }
    val dataPro = data.map {
      case (time: String, userID: Long, keyWord: String, rankNum: String, url: String) =>
        (time, userID, keyWord, rankNum, url)
    }
    dataPro.repartition(1).saveAsTextFile(outPutpath + "dataPro")
    val dataAsMap = dataPro.groupBy {
      case (time, userID, keyWord, rankNum, url) => keyWord
    }.collectAsMap

    //按关键词排序
    for ((k, v) <- dataAsMap.toSeq.sortBy(_._1)) {
      //按userID排序
      val value = v.toSeq
      val valueRDD = spark.parallelize(value)
      val outK = k
      var saveFileName = outK.replaceAll("[^(a-zA-Z0-9\\u4e00-\\u9fa5)]", "")
      //判断处理后的文件名是否合法
      if (saveFileName.isEmpty()) {
        saveFileName = "Null"
      }
      val saveFile = new PrintWriter(outPutpath + saveFileName)
      value.foreach {
        line =>
          saveFile.print(line)
          saveFile.println
      }
      saveFile.close

    }

  }

}