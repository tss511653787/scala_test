
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors

import java.util.Arrays
object bsk {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[4]", "Bsk")
    val path = "hdfs://tss.hadoop2-1:8020/user/root/dataSet/data/test2_bsk.csv"

    //使用第一行来定义一个headerclass
    class SimpleCSVHeader(header: Array[String])
        extends Serializable {
      val index = header.zipWithIndex.toMap
      def apply(array: Array[String],
                key: String): String = array(index(key))
    }
    //去数据头部
    //sed 1d test_2.csv > test_2_nohead.csv
    val rawdata = sc.textFile(path)
    //csv格式数据以英文,分割数据S
    val records = rawdata.map(line => line.split(",")).saveAsTextFile("hdfs://tss.hadoop2-1/tools/dataSet/jieguo/")
    
    
    
        
    
  }
}