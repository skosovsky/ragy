//go:build ignore
package main
import("context";"fmt";"time";"github.com/skosovsky/ragy/access";"github.com/skosovsky/ragy/filter";"github.com/skosovsky/ragy/lexical";"github.com/skosovsky/ragy/retrieval")
type meta struct{Tenant string `json:"tenant"`}
func main(){
 fields:=filter.NewSchema();tenant,_:=fields.String("tenant");schema,_:=fields.Build();builder,_:=filter.NewBuilder(schema); mandatory,_:=filter.Eq(builder,tenant,"a").Build()
 pub,_:=access.PinPublication("fixed",[]access.TargetRevision{{Target:"lexical",Namespace:"n",Source:"s",Revision:"r1",Transformation:"text",AccessFingerprint:"acl"}})
 now:=time.Unix(100,0);read,_:=access.Scoped(access.ScopedConfig{Schema:schema,Mandatory:mandatory,Publication:pub,Snapshot:access.Snapshot{Identity:"scope",PolicyEpoch:1,IssuedAt:now,ExpiresAt:now.Add(time.Minute)},Now:func()time.Time{return now},Authority:access.AuthorityFunc(func(context.Context,access.Snapshot)error{return nil})})
 snap,err:=lexical.NewBM25Snapshot(context.Background(),schema,lexical.Config[*meta]{SearchFields:[]string{"content"}},read,[]retrieval.Document[*meta]{{ID:"d",Content:"secret",Meta:&meta{Tenant:"a"}}},func(m *meta)(*meta,error){v:=*m;return &v,nil});if err!=nil{panic(err)}
 q:=retrieval.Query[struct{}]{Read:read,Text:"secret",Options:retrieval.RetrieveOptions{TopK:10}}
 first,err:=snap.Retrieve(context.Background(),q);if err!=nil{panic(err)};docs:=first.Documents();fmt.Println("first",first.Len(),"tenant",docs[0].Meta.Tenant);docs[0].Meta.Tenant="b"
 second,err:=snap.Retrieve(context.Background(),q);fmt.Println("after caller mutation",second.Len(),"error",err,"first batch changed",first.Documents()[0].Meta.Tenant)
}
