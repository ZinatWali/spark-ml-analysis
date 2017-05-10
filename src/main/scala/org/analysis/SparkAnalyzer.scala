package org.analysis

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

object SparkAnalyzer {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("MyApp")
      .getOrCreate()

    val df = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("s3n://data-bucket-phase-1/2014GSSNDI.csv")

      val trimmed = df
      .selectExpr(
      "wrkstat", "cast(hrs1 as int)", "cast(hrs2 as int) hrs2", "evwork", "wrkslf", "wrkgovt", "cast(commute as int)", "marital",
      "cast(agewed as int)", "spevwork", "cast(sphrs1 as int)", "cast(sphrs2 as int)",
      "cast(sibs as int)", "cast(childs as int)",
      "cast(educ as int)", "cast(paeduc as int)", "cast(maeduc as int)", "cast(speduc as int)",
      "cast(age as int)", "sex", "region", "race", "polviews", "polviewy",
      "happy")

    val cleaned = trimmed.na.fill(0)

    cleaned.printSchema()
    cleaned.select(cleaned("wrkstat"), cleaned("hrs1"), cleaned("age")).take(5).foreach(x => println(x))

    val catColumns = List("wrkstat", "evwork", "wrkslf", "wrkgovt", "marital", "spevwork", "sex", "region", "race", "polviews", "polviewy")

    val assemblerForIndexing = new VectorAssembler()
      .setInputCols(Array("hrs1", "hrs2", "agewed", "sphrs1", "sphrs2", "sibs", "childs", "educ", "paeduc", "maeduc", "speduc", "age"))
      .setOutputCol("NumericFeatures")

    val featureIndexer = new VectorIndexer()
      .setInputCol("NumericFeatures")
      .setOutputCol("NumericFeaturesIndexed")
      .setMaxCategories(4)

    val catIndexers = catColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_Index")
    )

    val encoders: List[OneHotEncoder] = catColumns.map(
      cname => new OneHotEncoder()
        .setInputCol(s"${cname}_Index")
        .setOutputCol(s"${cname}_Vec")
    )

    val assemblerForRF = new VectorAssembler()
      .setInputCols(("NumericFeaturesIndexed" :: catColumns.map(x => x + "_Vec")).toArray)
      .setOutputCol("Features")

    val labelIndexer = new StringIndexer()
      .setInputCol("happy")
      .setOutputCol("happyIndexed")

    val rf = new RandomForestClassifier()
      .setLabelCol("happyIndexed")
      .setFeaturesCol("Features")

    val processPipe = new Pipeline().setStages((List(assemblerForIndexing, featureIndexer) ++ catIndexers ++ encoders ++ List(assemblerForRF, labelIndexer, rf)).toArray)

    val Array(trainingData, testData) = cleaned.randomSplit(Array(0.7, 0.3))

    val model = processPipe.fit(trainingData)

    println("Model fitting completed")

    val predictions = model.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("happyIndexed")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)

    println("Test Error = " + (1.0 - accuracy))

    spark.stop()
  }
}
