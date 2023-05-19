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
    let run kind (expFac:SweepablePipeline -> AutoMLExperiment) (g:Grammar)=
        let genomSize = Grammar.esimateGenomeSize g
        let parms = genomSize |> List.map(fun x -> I(0,0,x)) |> List.toArray

        let mutable fmap = Map.empty
        let fitness (pvals:float[]) = 
            let genome = pvals |> Array.map int
            fmap
            |> Map.tryFind genome
            |> Option.defaultWith(fun _ -> 
                let terminals,_ = Grammar.translate g genome
                let pipeline = Grammar.toPipeline terminals
                let exp = expFac pipeline                
                let rslt = exp.Run()
                fmap <- fmap |> Map.add genome rslt.Metric
                rslt.Metric
            )
             
        let mutable step = CALib.API.initCA(parms, fitness, kind,36)
        for i in 0 .. 15000 do 
            step <- CALib.API.Step step
        
        let gbest = step.Best.[0].MParms |> Array.map int
        let gbestTerms,_ = Grammar.translate g gbest 
        gbestTerms, step.Best.[0].MFitness
