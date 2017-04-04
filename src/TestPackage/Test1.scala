

object Test1 {
  def main(args: Array[String]) {
    val f1 = mubly(2)
    val f2 = mubly(5)
    println(f1(5))
    println(f2(5))
    println(mul(3)(4))
    val list = List(2, 3, 4, 5)
    //折叠
    val sum = list.fold(1)((x, y) => x + y)
    println(sum)
    println(list.sortWith(_ > _))

  }
  //闭包
  def mubly(factor: Double) = (x: Double) => x * factor
  //柯里化
  def mul(x: Int) = (y: Int) => x * y
  def multply(x: Int)(y: Int) = x * y

}