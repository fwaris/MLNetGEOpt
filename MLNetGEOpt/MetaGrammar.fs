namespace MLNetGEOpt
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLUtils.Pipeline

//meta grammar to describe the grammar for the grammatical evolution process
//
(*
What do we need?
Given:
- linear genome composed of float values
- genome (genotype) translated to AST (phenotype)
- AST evaluated for fitness
- population of linear genome evolved via 'search engine'
Grammar:
- start, non-terminals, terminals [keywords | [value (domain / range)]], production rules
*)

type Term = 
    | Opt of Term 
    | Pipeline of SweepablePipeline 
    | Estimator of SweepableEstimator  
    | Alt of Term list
    | Union of Term list

type Grammar = Term list

type Srch =
    static member Double(s:string,lo:float,hi:float,?isLog) = let ss = SearchSpace() in ss.Add(s,Option.UniformDoubleOption(lo,hi,?logBase=isLog)); ss

type Fac1 = 
    static member  fac1 (ctx:MLContext) : IEstimator<ITransformer> = ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("a",labelColumnName="b") :> IEstimator<ITransformer>
    

module Exp =

    let rec tranlateTerm (genome:int[]) (acc,i) (t:Term) =
        let i = if i >= genome.Length then 0 else i
        match t with
        | Pipeline _ | Estimator _ -> (t::acc),i 
        | Opt t'  -> 
            if genome.[i] = 0 then 
                tranlateTerm genome (acc,i+1) t'
            else
                acc,(i+1)
        | Alt ts ->
            let t' = ts.[genome.[i]]     
            tranlateTerm genome (acc,i+1) t'
        | Union ts -> 
            match ts with 
            | [] -> acc,i
            | t::rest -> 
                let acc,i = tranlateTerm genome (acc,i) t
                tranlateTerm genome (acc,i+1) (Union rest)

    let translate genome grammar =
        (([],0),grammar) ||> List.fold (fun (acc,i) t -> tranlateTerm genome (acc,i) t)

    let toPipeline ts = 
        let h,ts =
            match ts with
            | Pipeline p::rest -> p,rest
            | (Estimator e1)::(Estimator e2)::rest -> e1.Append(e2),rest
            | (Estimator e1)::(Pipeline e2)::rest  -> e1.Append(e2),rest
            | _                                    -> failwith "Given list should be only Pipeline or Estimator terms with atleast two estimators or one pipeline"
        (h,ts) 
        ||> List.fold (fun acc t ->
            match t with 
            | Estimator e -> acc.Append(e)
            | Pipeline p -> (acc,p.Estimators) ||> Seq.fold(fun acc kv -> acc.Append(kv.Value))
            | _          -> failwith "Only Pipeline or Estimator terms expected. Ensure 'translate' is called")


