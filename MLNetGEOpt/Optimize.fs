namespace MLNetGEOpt
open CA
open Microsoft.ML.AutoML
open Microsoft.ML

type Exp<'a when 'a : not struct > = 
    | Predefined of ExperimentBase<'a,ExperimentSettings> * IDataView * IDataView 
    | Custom of AutoMLExperiment

module Optimize =
    (*
        evolved the genome using CA rules
        convert CA individual to sweepable pipeline
        evaluate sweepable pipeline
        use cache to speed up processing
        feedback fitness to CA for next gen evolution

        What is the fitness function here?
        We leverage AutoMLExperiment. We need:
        - dataset
        - sweeplable pipeline (see above)
        - trialRunner
        - metric 
          - binary classification
          - multi-class classfication
          - regression         
    *)
    let opt<'a when 'a: not struct> (g:Grammar) (expFac:unit -> Exp<'a>) =
        let genomSize = Grammar.esimateGenomeSize g
        let parms = genomSize |> List.map(fun x -> I(0,0,x)) |> List.toArray

        let fitness (pvals:float[]) = 
            let genome = pvals |> Array.map int
            let terminals = Grammar.translate genome g
            let pipeline,_ = Grammar.toPipeline terminals
            let exp = expFac()
            match exp with
            | Predefined (exp,_,_) -> exp.Execute()
            


    let ctx = MLContext()
    let exp = ctx.Auto().CreateBinaryClassificationExperiment(39u)
    let oset = AutoMLExperiment.AutoMLExperimentSettings()
    oset.SearchSpace <- exp.
    let oexp = AutoMLExperiment(ctx,oset)
    let b1 = BinaryClassificationExperiment()
    let ex2 = exp :?> AutoMLExperiment

    let exp1 = AutoMLExperiment()


    let o =
        {new ITrialRunner with
             member this.Dispose() = raise (System.NotImplementedException())
             member this.RunAsync(settings, ct) = 
                task {
                    return 
                        TrialResult(
                            Metric = 0.,
                            Model = null,
                            TrialSettings = settings,
                            DurationInMilliseconds = 0.0
                        )
                }
                
        }

    ()

