package HotalDataHotWords

import java.util.ArrayList

object Test {
  def main(args: Array[String]): Unit = {
    val arr = Array("a", "b", "c", "d", "d", "e", "f", "f", "g")
    val findwords = "a"
    val answer = new ArrayList[String]()
    for (i <- 0 to arr.length - 1) {
      var temp = new StringBuilder()
      if (arr(i).equals(findwords)) {
        temp ++= "(" + arr(i) + ")"
        //找到词
        var k = i
        //将前面4个人词加入到temp
        var m = 0
        while (((k - 1) != -1) && (m != 4)) {
          temp.insert(0, arr(k - 1) + " ")
          m = m + 1
          k = k - 1
        }
        //将后面4个加入到temp
        temp.append(" ")
        k = i
        var n = 0
        while (((k + 1) != arr.length) && (n != 4)) {
          temp.append(arr(k + 1) + " ")
          n = n + 1
          k = k + 1
        }
        //加入到answer
        answer.add(temp.toString())
      }
    }
    val res = answer.toArray()
    res.foreach(println)
  }
}