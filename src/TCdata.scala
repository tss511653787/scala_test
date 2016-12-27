import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer

class TCdata {
  /*
   * 数据表
   * 对数据表中某一个属性列中 多个数据进行wordcount计数
   * */
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("TCdata")
    val spark = new SparkContext(conf)
    //引入spark sql标签
    val sqlContext = new org.apache.spark.sql.SQLContext(spark)
    import sqlContext.implicits._
    //读取数据
    val inputpath = "C:/Users/dell/Desktop/data/TCdata.csv"
    val outputpath = "C:/Users/dell/Desktop/TCoutput/"
    val src = spark.textFile(inputpath)
    //提取列PN
    val colPN = src.map { line =>
      val spl = line.split(",")
      spl(0)
    }
    colPN.cache()
    colPN.repartition(1).saveAsTextFile(outputpath + "colPN")
    //处理
    val colPNres = colPN.map { line =>
      //分隔符:";  "
      val sp = line.split(";  ")
      //截取前2个字母
      val cp = sp.map { str => str.substring(0, 2)
      }
      cp
    }
    val savepath = "C:/Users/dell/Desktop/TCoutput/"
    val save = new PrintWriter(savepath + "save")
    val PNarr = colPNres.collect
    PNarr.foreach { arr =>
      //Array数组转化为RDD
      val arrRDD = spark.parallelize(arr)
      val res = arrRDD.map { x => (x, 1) }.reduceByKey(_ + _)
      //对wordcount结果进行格式化输出
      val colres = res.collect
      colres.foreach { line =>
        save.print(line + " ")
      }
      save.println
    }
    save.close

  }

}