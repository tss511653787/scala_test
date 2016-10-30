
object WS1 {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(57); 
  println("Welcome to the Scala worksheet");$skip(58); 
  val str1=Array("apple a d f ","b n j k","a b cccc d d");System.out.println("""str1  : Array[String] = """ + $show(str1 ));$skip(36); 
 val str2=str1.map(x=>x.split(" "));System.out.println("""str2  : Array[Array[String]] = """ + $show(str2 ));$skip(54); 
 val str3=str1.flatMap(x=>x.split(" ")).map(x=>(x,1));System.out.println("""str3  : Array[(String, Int)] = """ + $show(str3 ))}
  
}
