package data_trans_test

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.sql._
import utils.AnaylyzerTools
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.log4j.Level
import org.apache.log4j.Logger
import java.io.PrintWriter
import java.util.ArrayList
import scala.collection.mutable.ArrayBuffer

object Copy_testdata {
  //屏蔽日志
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    val conf = new SparkConf()
      //***这个地方必须是local***
      .setMaster("local")
      .setAppName("testdata")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //建立cherkpoint点
    val cherkPointPath = "C:/Users/Administrator/Desktop/cherkpoint"
    SaveFile.makeDir(cherkPointPath)
    sc.setCheckpointDir(cherkPointPath)

    //引入隐式转换
    import sqlContext.implicits._


    //文件路径 
    val path = "F:/data/Car_data/Cardata5000.csv"
    //读取文件
    val text = sc.textFile(path)
    text.cache()
    text.checkpoint();

    //将元数据转化为DataFrame
    val data = text.map {
      rawline =>
        val splitdata = rawline.split(",")
        CopyDataRecord(splitdata(0), splitdata(1), splitdata(2), splitdata(3), splitdata(4), splitdata(5).toInt, splitdata(6))
    }
    val dataDF = data.toDF
    dataDF.cache
    dataDF.show

    //dataDF的本地持久化
    val newpath = "C:/Users/Administrator/Desktop/ouput/"
    dataDF.rdd.repartition(1).saveAsTextFile(newpath + "dataDF")

    /*
     * 数据预处理阶段
     * 分词
     * 去停用词
     * 过滤噪音
     * 整理格式
     */

    //分词处理
    val casetext = dataDF.rdd.map {
      case Row(title: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
        text
    }
    //为每个条数建立索引序号(1)
    var indexnum = 0
    val caseall = dataDF.rdd.map {
      case Row(title: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
        indexnum = indexnum + 1
        (indexnum, title, content, text, name, time, recall, net)
    }
    val caseallDF = caseall.toDF
    caseallDF.cache
    caseallDF.show
    //重置列名属性
    val recaseDF = caseallDF.select(caseallDF("_1").as("index"), caseallDF("_2").as("title"), caseallDF("_3").as("content"), caseallDF("_4").as("text"), caseallDF("_5").as("name"), caseallDF("_6").as("time"), caseallDF("_7").as("recall"), caseallDF("_8").as("net"))
    recaseDF.cache

    //对数据进行切分和简单热度统计
    val highrecall = recaseDF.where("recall>=100")
    println("回帖数量超过100的帖子有:" + highrecall.count())
    highrecall.rdd.repartition(1).saveAsTextFile(newpath + "highrecallRDD")
    val arrtext = casetext.collect
    val savechar = new PrintWriter(newpath + "savechar")
    arrtext.foreach { line => savechar.println(line) }
    //使用javaIO输出必须关闭流否则会被释放掉 切记！！！
    savechar.close

    //运行分词类
    CopyOfAnaylyzerTools.split()
    //另一个想法
    /* 由于无法序列化分词包中的方法目前分词暂时无法集成在Spark中处理
     * 分词无法并行化
        val anaylyzerTools = new CopyOfAnaylyzerTools()
        val configg = anaylyzerTools.config
        val dicc = anaylyzerTools.dic
        val splitText = casetext.map {
          line =>
            val list = anaylyzerTools.anaylyzerWords(line, configg, dicc)
            list.toString()
              .replace("[", "").replace("]", "")
              .replaceAll(" ", "").replaceAll(",", " ")
              .replaceAll("[(]", "").replaceAll("[)]", "")
        }
        splitText.repartition(1).saveAsTextFile(newpath + "splitText")
        */

    //再次读入结果
    val inputpath = "C:/Users/Administrator/Desktop/splitoutput/*"
    val wordsDS = sc.wholeTextFiles(inputpath)
    wordsDS.cache
    val toint = wordsDS.map {
      case (file, text) => (file.split("/")(6).toInt, text)
    }
    toint.cache
    val tointDF = toint.toDF()
    tointDF.cache
    tointDF.show
    val renamedata = tointDF.select(tointDF("_1").as("index"), tointDF("_2").as("splitwords"))
    renamedata.cache

    //将持久化的分词结果和元数据进行笛卡尔积
    val datajoinDF = renamedata.join(recaseDF, renamedata("index") === recaseDF("index"))
    datajoinDF.cache
    val DataDF = datajoinDF
    DataDF.cache
    DataDF.show
    DataDF.rdd.repartition(1).saveAsTextFile(newpath + "ddataDF")

    //显示参数
    DataDF.explain()

    //spword RDD
    //格式:[index,分词，时间，回复数]
    val spword = DataDF.rdd.map {
      case Row(index: Int, splitwords: String, aindex: Int, titile: String, content: String, text: String, name: String, time: String, recall: Int, net: String) =>
        (index, splitwords, time, recall)
    }
    spword.cache()
    spword.repartition(1).saveAsTextFile(newpath + "spword")
    val numword = spword.map {
      case (index, splitwords, time, recall) =>
        splitwords
    }
    numword.cache
    //统计去重的单词数目
    val num = numword.flatMap(_.split(" "))
    val number = num.distinct().count()
    println("分词去重统计：" + number)
    val outdata = spword.map(line => {
      val str = line.toString
      val reback = str.replace("[", "").replace("]", "").replace("[(]", "").replace("[)]", "")
      reback
    })
    outdata.cache
    //输出预处理结果
    outdata.repartition(1).saveAsTextFile(newpath + "outdata")
    /* 接下来的工作：
     * 2016/10/15
     * output
     * rdd存储下来有使用notepad替换掉'('和')'
     * 转换成utf-8格式作为下一步输入
     * 
     * */
  }
}
case class CopyDataRecord(title: String, content: String, text: String, name: String, time: String, recall: Int, Net: String)
