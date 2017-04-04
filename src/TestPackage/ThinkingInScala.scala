import scala.collection.mutable.ListBuffer
import scala.util.Random
import scala.collection.mutable.HashMap

object ThinkingInScala {
  //方法和函数的区别
  val f = (x: Int, y: Int) => x + y
  val p = (m: Int, n: Int) => {
    val result = m * n * n
    result
  }
  def m(f: (Int, Int) => Int) = {
    f(2, 2)
  }

  def find(s: Int, name: String): String = {
    name + s.toString
  }
  //神奇
  val finding: (Int, String) => String = { (num, name) => name.toString }
  //将方法转变成函数
  val f2 = m _

  //样例类
  case class subminttask(id: String, name: String)
  case class heart(time: String)
  case class cherktime(id: String, name: String, age: Int)

  def main(args: Array[String]) {
    //样例类模式匹配
    val arrAll = Array(cherktime("tss", "tss", 25), subminttask("ssss", "tttt"), heart("ggggg"), heart("abccc"))
    arrAll(Random.nextInt(arrAll.length)) match {
      case cherktime(id, name, age) => {
        println(s"$id $name $age")
      }
      case subminttask(id, name) => {
        println(s"$id $name")
      }
      case heart(time) => {
        println(time)
      }

    }
    val arr = Array("abc", "b", 123, 0.25)
    val num = Array(1, 2, 3, 4)
    val numWithArr = num.zip(num)
    val toMap = numWithArr.toMap
    val res = toMap.getOrElse(6, 0)
    //练习 单机版WC
    //1分部练习
    val list = List("apple", "apple", "tss", "zst", "zst", "banana", "banana")
    val mapList = list.map { word => (word, 1) }
    //groupedBy返回值是一个Map类型
    val groupedMap = mapList.groupBy(_._1)
    val resGroup = groupedMap.map { map => (map._1, map._2.size) }
    val reslist = resGroup.toList
    val resWC = reslist.sortBy(_._2)
    println(resWC)
    //一体化
    val rres = list.map { x => (x, 1) }.groupBy(_._1).map(x => (x._1, x._2.length)).toList.sortBy(_._2)
    println(rres)
    //HashMap方法
    val hash = new HashMap[String, Int]
    list.foreach { word =>
      hash(word) = hash.getOrElse(word, 0) + 1
    }
    println(hash.toList.sortBy(_._2))
  }

}