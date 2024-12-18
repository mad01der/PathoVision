/*
 * @Date: 2022-04-17 17:39:09
 * @LastEditors: JZY
 * @LastEditTime: 2023-01-02 18:57:53
 * @FilePath: /project/Visual/src/App.jsx
 */
// import "./App.css";
import "./home_load.css";
import React, { Component } from "react";
import { Row, Layout, Space, Col, Spin, Alert,Button ,Modal,Progress} from "antd";
import axios from "axios";
// 导入组件
import CoreModule_fix from "../CoreModule_fix";
import MapVision_fix from "../MapVision_fix";
import LeftModule from "../LeftModule";
import TopModule_fix from "../TopModule_fix";
import PicModule_fix from "../PicModule_fix/Middle";
import CropModule from "../CropModule"
const { Content, Header } = Layout;
const imageUrl = "C:/Users/李金阳/Desktop/TVCG2/Visual/public/OIP.jpg"
export default class Home extends Component {
  constructor(props) {
    super(props);
    this.state = {
      alertVisible: false,
      alertMessage: "",
      progress:0,
      choosePatches: [0],
      chooseMapImg: 0,
      mapValid: true,
      init_loading: 0,
      alert: {
        message: "Initial Progress",
        description:
          "Now, we have such 3 stage to do. First,we load the dataset.Then we pre-training model. Finally, we need generate AL result for 1th iteration...",
        type: "info",
      },
      epoch_Data: {},
      sample_Data: {},
      WSI_Data: {},
      New_data:{},
      confusion_Data: {},
      bk_data: {},
      
      current_iteration: 0,
    };
  }
  




  
  componentDidMount = () => {
    var T = this;
    console.log("init...")
    axios
      .get("http://127.0.0.1:5000/init")
      .then(function (response) {
        if (response.data.load_status == 200) {
          T.setState({
            init_loading: 1,
            epoch_Data: response.data.epoch_Data,
            sample_Data: response.data.sample_data,
            WSI_Data: response.data.WSI_Data,
            confusion_Data: response.data.confusion_Data,
            New_data: response.data.New_data,
            bk_data: response.data.bk_data,
            current_iteration: response.data.iteration,
          });
        } else {
          T.setState({
            alert: {
              message: "Something error!",
              description:
                "We have a problem in initial stage, Please check your dataset...",
              type: "error",
            },
            init_loading: -1,
          });
        }
      })
      .catch((err) => {
        T.setState({
          alert: {
            message: "Connection error!",
            description:
              "We can't connect the data, Please check your network connection...",
            type: "error",
          },
          init_loading: -1,
        });
      });
  };
  changeDeletePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.coreModuleRef.changeDeletePatches(p);
  };

  


  




 
  changeChoosePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.mapChildRef.changeChoosePatches(p);
  };
  showMap = async (img_id) => {
    await this.setState({
      mapValid: true,
      chooseMapImg: img_id,
    });
    this.mapChildRef.drawChart();
  };

  closeMap = () => {

    this.setState({
      mapValid: false,
    });
  };

  handleMapChildEvent = (ref) => {
    this.mapChildRef = ref;
  };
  handleCoreModuleEvent = (ref) => {
    this.coreModuleRef = ref;
  };
//   train = () => {
//     var T = this;
//       this.setState({
//         // alert: {
//         //   message: "start training!",
//         // },
//         // init_loading: 0,
//         alertVisible: true,
//         alertMessage: "loading...",
//       });
//       const interval = setInterval(() => {
//         const progress = this.state.progress + 1;
//         this.setState({ progress });

//         // 根据不同的进度阶段设置不同的提示信息
//         if (progress === 10) {
//             this.setState({ alertMessage: "Finish processing" });
//         } else if (progress === 20) {
//             this.setState({ alertMessage: "Start training" });
//         } else if (progress === 90) {
//             this.setState({ alertMessage: "Active learning epoch" });
//         } else if (progress === 100) {
//             this.setState({ alertMessage: "Training completed" });
//             clearInterval(interval); // 停止定时器
//         }
//     }, 600); // 更新频率为每秒更新一次
    
//     axios
//       .post("http://127.0.0.1:5000/train")
//       .then(function (response) {
//         if (response.data.load_status == 200) {
//           T.setState({
//             init_loading: 1,
//             alert: {
//               message: "train finished !",
//             },
//           });
//         } else {
//           T.setState({
//             alert: {
//               message: "Something error!",
//               description:
//                 "We have a problem in train stage, Please check your dataset...",
//               type: "error",
//             },
//             init_loading: -1,
//           });
//         }
//       })
//       .catch((err) => {
//         T.setState({
//           alert: {
//             message: " error!",
//             description:
//               "We have a problem in train stage, Please check your dataset...",
//             type: "error",
//           },
//           init_loading: -1,
//         });
//       });
// }
  render() {
    
    return (
      <>
        <Layout>
          {this.state.init_loading != 1 ? (
            <div id="example">
              <Spin spinning={this.state.init_loading == 0}>
                <Alert
                  message={this.state.alert.message}
                  description={this.state.alert.description}
                  type={this.state.alert.type}
                  showIcon
                />
              </Spin>
            </div>
          ) : (
            <>
              <Header className="headerModule">
                <TopModule_fix current_iteration={this.state.current_iteration}
                           WSI_Data={this.state.WSI_Data} 
                /> 
                {/* <div >
                   
                   <Col>
                      <Button type="primary" shape="round" size="small" onClick={this.train}>
                          Upload and train new datasets
                      </Button>
                  </Col>
                <Modal
                   title="Now training"
                   visible={this.state.alertVisible}
                   onCancel={() => this.setState({ alertVisible: false })}
                   footer={null}
                >
                   <p>{this.state.alertMessage}</p>
                   <Progress percent={this.state.progress} status="active" />
               </Modal> 
               </div> */}
                {/* <Col offset={1}>
                 <Button type="primary" shape="round" size="small" onClick={this.check_new_result}>
                  check new results
                </Button>
               </Col> 
               <Col offset={1}>
                 <Button type="primary" shape="round" size="small" onClick={this.check_example_result}>
                  check example results
                 </Button>
               </Col> */}
              </Header>
              <Content className="site-layout">
                <Space className="basic" direction="vertical" size="small">
                  <Row gutter={[5, 5]} justify="space-around" id="one">
                    <Col span={4}>
                      <LeftModule
                        epoch_Data={this.state.epoch_Data}
                        current_iteration={this.state.current_iteration}
                        confusion_Data={this.state.confusion_Data}
                      />
                    </Col>
                    <Col span={3} >
                      <PicModule_fix
                          WSI_Data={this.state.WSI_Data}
                          changeChoosePatches={this.changeChoosePatches}
                          choosePatches={this.state.choosePatches}
                      />
                    </Col>
                    <Col span={9} id="mapVision">
                      <MapVision_fix
                          onChildEvent={this.handleMapChildEvent}
                          ref={this.mapChildRef}
                          changeChoosePatches={this.changeChoosePatches}
                          showMap={this.showMap}
                          chooseMapImg={this.state.chooseMapImg}
                          changeDeletePatches={this.changeDeletePatches}
                          choosePatches={this.state.choosePatches}
                          epoch_Data={this.state.epoch_Data}
                          sample_Data={this.state.sample_Data}
                          WSI_Data={this.state.WSI_Data}
                          bk_data={this.state.bk_data}
                          New_data = {this.state.New_data}
                          current_iteration={this.state.current_iteration}
                      />
                    </Col>
                    <Col span={8} id="mainMap">
                      <CoreModule_fix
                        chooseMapImg={this.state.chooseMapImg}
                        onChildEvent={this.handleCoreModuleEvent}
                        ref={this.coreModuleRef}
                        changeChoosePatches={this.changeChoosePatches}
                        choosePatches={this.state.choosePatches}
                        mapValid={this.state.mapValid}
                        epoch_Data={this.state.epoch_Data}
                        sample_Data={this.state.sample_Data}
                        WSI_Data={this.state.WSI_Data}
                        bk_data={this.state.bk_data}
                        New_data = {this.state.New_data}
                        current_iteration={this.state.current_iteration}
                      />
                    </Col>
                  </Row>
                </Space>
              </Content>
            </>
          )}
        </Layout>
      </>
    );
  }
}
