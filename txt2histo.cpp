void txt2histo(){
    //usage: 
    // input txtdata must be separated by space
    // 0 324.43 635.2
    // 1 4324.32 56.24
    // 2 542.53 673.9
    // ChannelOfInterest determin which channel will be filled into histo

//read data value
  std::string filename = "../E16SIM/bkg_ee_betaGamma.txt";
  const int ChannelOfInterest = 1;
  
  std::ifstream file(filename);
  std::string line;
  std::vector<std::vector<double>> h;
  while (std::getline(file, line)) {
    double value;
    std::stringstream ss(line);
    h.push_back(std::vector<double>());
    while (ss >> value) {
      h[h.size()-1].push_back(value);
    }
  }

 //fill histgram 

  int max_x = 1000;
  int min_x = 0;
  int nBin = (max_x - min_x) / 10;

  auto hist = new TH1F("test",Form("%s ch = %d",filename.c_str(),ChannelOfInterest),nBin,min_x,max_x); 

  for (int i = 0; i < h.size(); i++) {

    hist->Fill(h[i].at(ChannelOfInterest));
  
}

  hist->Draw();

}