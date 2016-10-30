
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy

object fenlei_test {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(1824); 
  def main(arg: Array[String]): Unit = {
    //初始化Sparcontextest
    val sc = new SparkContext("local[2]", "Fen lei text") //本地模式
    // sed 1d train.tsv > train_noheader.tsv
    // load raw data 去掉第一行列标签

    val rawData = sc.textFile("kaggle/train_noheader.tsv")
    val records = rawData.map(line => line.split("\t"))
    println("结果输出1：")
    records.first.foreach(x => (print(x)))
    //去掉数据中的“ 将数据中的？用0补全
    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))

    }
    data.cache
    val numData = data.count()
    println(numData)
    //为朴素贝叶斯整理数据，特征值非负
    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }
    //训练分类模型
    //为逻辑回归和SVM设置最大迭代次数 为决策树设置最大深度
    val numIterations = 10
    val maxTreeDepth = 5
    //训练逻辑回归模型
    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

  };System.out.println("""main: (arg: Array[String])Unit""")}

}
