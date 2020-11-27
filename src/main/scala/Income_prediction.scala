import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._

import scala.io.Source
import java.nio.charset.CodingErrorAction

import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.functions._
import org.apache.spark.ml._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.PCA



object Income_prediction {

  val spark : SparkSession = SparkSession.builder
    .appName("income_study")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._


  // Set the log level to only print errors
  Logger.getLogger("org").setLevel(Level.ERROR)


  def main(args: Array[String]): Unit = {

    val classification = false
    val model_choice = "LR" // Logistic Regression (LR) or Random Forest (RF)
    val clustering = true



    // DATA PREPARATION

    // load data to dataframe
    val data = spark.read
      .format("csv")
      .option("delimiter", ",")
      // .option("header", "true")
      .load("income_data.csv")
      .toDF("age_vec", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
        "race", "sex", "capital-gain", "capital-loss", "hours-per-week_vec", "native-country", ">50K")
      .withColumn("age_vec", $"age_vec".cast(sql.types.FloatType))
      .withColumn("hours-per-week_vec", $"hours-per-week_vec".cast(sql.types.FloatType))
      .na.drop()
      .distinct()


    // label the incomes
    val indexer = new StringIndexer()
      .setInputCol(">50K")
      .setOutputCol("label")
    val data_label = indexer.fit(data).transform(data)


    // One Hot encode the categorical features
    val features = data_label.columns.filterNot(_.contains("id"))
      .filterNot(_.contains("education-num"))
      .filterNot(_.contains("age_vec"))
      .filterNot(_.contains("capital-gain"))
      .filterNot(_.contains("capital-loss"))
      .filterNot(_.contains("hours-per-week_vec"))
      .filterNot(_.contains("label"))
      .filterNot(_.contains(">50K"))
      .filterNot(_.contains("fnlwgt"))

    val encodedFeatures = features.flatMap { name =>
      val stringIndexer = new StringIndexer()
        .setInputCol(name)
        .setOutputCol(name + "_Index")

      val oneHotEncoder = new OneHotEncoderEstimator()
        .setInputCols(Array(name + "_Index"))
        .setOutputCols(Array(name + "_vec"))
        .setDropLast(false)

      Array(stringIndexer, oneHotEncoder)
    }
    val pipeline = new Pipeline().setStages(encodedFeatures)
    val indexer_model = pipeline.fit(data_label)
    val data_encoded = indexer_model.transform(data_label)


    // using vector assembler to combine the features created into a single feature vector
    val vecFeatures = data_encoded.columns.filter(_.contains("vec")) // the features used to create the final feature vector ends with _vec
    val vectorAssembler = new VectorAssembler()
      .setInputCols(vecFeatures)
      .setOutputCol("features")

    val pipelineVectorAssembler = new Pipeline().setStages(Array(vectorAssembler))

    val data_feature = pipelineVectorAssembler.fit(data_encoded).transform(data_encoded)
      .select("features", "label")


    // CLASSIFICATION

    if (classification) {


      val Array(training, test) = data_feature.randomSplit(Array(0.8, 0.2), seed = 42)


      // Search for the best model's hyperparameters

      if (model_choice == "RF") {
        val rf = new RandomForestClassifier()
          .setLabelCol("label")
          .setFeaturesCol("features")

        val paramGrid = new ParamGridBuilder()
          .addGrid(rf.maxDepth, Array(10, 15, 20))
          .addGrid(rf.numTrees, Array(40, 60, 80, 100))
          .build()

        val cv = new CrossValidator()
          .setEstimator(rf)
          .setEvaluator(new BinaryClassificationEvaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(3)
          .setParallelism(2)

        val evaluator = new BinaryClassificationEvaluator()
          .setMetricName("areaUnderPR")

        val RfModel = cv.fit(training)
        val bestRrModel = RfModel.bestModel

        println("Best hyperparameters founds:")
        println(bestRrModel.extractParamMap())

        val result = bestRrModel.transform(test)
        println(">>> Test set Area Under PR Curve for Random Forest = " + evaluator.evaluate(result))

      }

      if (model_choice == "LR") {
        val lr = new LogisticRegression()
          .setMaxIter(10)
          .setLabelCol("label")
          .setFeaturesCol("features")

        val paramGrid = new ParamGridBuilder()
          .addGrid(lr.regParam, Array(0.20, 0.25, 0.30, 0.35))
          .addGrid(lr.elasticNetParam, Array(0.4, 0.5, 0.6, 0.7))
          .build()

        val cv = new CrossValidator()
          .setEstimator(lr)
          .setEvaluator(new BinaryClassificationEvaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(3)
          .setParallelism(4)

        val evaluator = new BinaryClassificationEvaluator()
          .setMetricName("areaUnderPR")

        val lrModel = cv.fit(training)
        val bestLrModel = lrModel.bestModel

        println("Best hyperparameters founds:")
        println(bestLrModel.extractParamMap())

        val result = bestLrModel.transform(test)
        println(">>> Test set Area Under PR Curve for Logistic Regression = " + evaluator.evaluate(result))

      }
    }


    // CLUSTERING
    if (clustering) {

      val pca = new PCA()
        .setInputCol("features")
        .setOutputCol("pca_features")
        .setK(3)
        .fit(data_feature)

      val data_pca = pca.transform(data_feature).select("pca_features")

      val evaluator = new ClusteringEvaluator()
        .setFeaturesCol("pca_features")
        .setPredictionCol("cluster")
        .setMetricName("silhouette")

      var best_k = 0
      var best_score = -1.0
      for (k <- 2 to 8) {

        val kmeans = new KMeans().setK(k).setSeed(42).setFeaturesCol("pca_features").setPredictionCol("cluster")
        val test_model = kmeans.fit(data_pca)
        val data_cluster = test_model.transform(data_pca)
        val score = evaluator.evaluate(data_cluster)
        println(k, score, test_model.computeCost(data_cluster))

        if (best_score < score) {
          best_score = score
          best_k = k
        }
      }

      println("Best k found : k=" + best_k.toString + " for a silhouette score of " + best_score.toString)
      val kmeans = new KMeans().setK(best_k).setSeed(42).setFeaturesCol("pca_features").setPredictionCol("cluster")
      val test_model = kmeans.fit(data_pca)
      val data_cluster = test_model.transform(data_pca)


      // A UDF to convert the pca_featurees from Vector to ArrayType
      val vecToArray = udf((xs: linalg.Vector) => xs.toArray)

      val data_arr = data_cluster.withColumn("pca_features_arr", vecToArray($"pca_features"))

      val data_csv = data_arr.withColumn("X", $"pca_features_arr".getItem(0))
        .withColumn("Y", $"pca_features_arr".getItem(1))
        .withColumn("Z", $"pca_features_arr".getItem(0))
        .select("X", "Y", "Z", "cluster")
      data_csv.show()

      data_csv.coalesce(1)
        .write
        .option("header", "true")
        .option("sep", ",")
        .mode("overwrite")
        .csv("clustered_data.csv")
    }
  }

}
