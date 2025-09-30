// cpu/baseline.cpp
#include <vector>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>

int main(int argc, char** argv){
  if(argc<3){ std::cerr<<"usage: baseline <bin> <shape>\n"; return 1; }
  int N,M; { std::ifstream s(argv[2]); s>>N>>M; }
  std::vector<uint8_t> seqs(N*M);
  std::ifstream f(argv[1], std::ios::binary); f.read((char*)seqs.data(), seqs.size());

  std::vector<float> D(N*N, 0.0f);
  for(int i=0;i<N;i++){
    D[i*N + i] = 0.0f;
    for(int j=i+1;j<N;j++){
      int mism=0, valid=0;
      const uint8_t* a = &seqs[i*M];
      const uint8_t* b = &seqs[j*M];
      for(int k=0;k<M;k++){
        uint8_t x=a[k], y=b[k];
        if(x==255 || y==255) continue;   // ignore gaps
        valid++;
        mism += (x!=y);
      }
      float pd = valid? (float)mism/(float)valid : 0.0f;
      D[i*N+j]=D[j*N+i]=pd;
    }
  }
  // dump result
  FILE* out=fopen("cpu_dist.bin","wb");
  fwrite(D.data(), sizeof(float), D.size(), out);
  fclose(out);
  std::cerr<<"CPU done\n";
}
