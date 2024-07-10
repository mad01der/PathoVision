import React, { Component } from "react";
import * as d3 from "d3";
import ReactECharts from "echarts-for-react";
import axios from "axios";

import {
  RedoOutlined,
  CloseOutlined,
  HeatMapOutlined,
  AimOutlined,
  MenuOutlined,
  AppstoreOutlined,
  RadarChartOutlined
} from "@ant-design/icons";
import {
  Button,
  Modal,
  Row,
  Col,
  Typography,
  Radio,
  Empty,
  Select,
  message,
  Tooltip,
  Tag,
  Image,
  Descriptions,
  Card,
  Spin,
  Drawer,
} from "antd";
import ShowImg from "./ShowImg";
const { Option } = Select;
// 常量设置
const { Title, Text } = Typography;
const xScale = d3.scaleLinear();
const yScale = d3.scaleLinear();
const maxValue = 30; // 缩放大小
const lineWidth = 0.1; // 分割线宽度
const rows = 65; //每行个数
const cols = 50; //每列个数
const imgSize = 30; //图片大小

const mapColors = [
  "#e7f1ff",
  "#b3d2ed",
  "#5ca6d4",
  "#1970ba",
  "#0c3b80",
  "#042950",
];
const group_color = [
  "#6fe214",
  "#2e2b7c",
  "#c94578",
  "#ebe62b",
  "#f69540",
  "#9f9a9a", // back
];
const mapLevel = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"];
const category_name = ["无癌症", "癌症", "高癌症趋势"];
const category_color = ["#40b373", "#d4b446", "#ee8826"]; // and more
const noise_tag_color = ["#90b7b7", "#a9906c", "#ef9a9a"];
const noise_tag_name = ["easy", "normal", "hard"];
const color_tag_name = ["label1: no cancer", "label2: cancer", "label3: high cancer"];
const color_tag_color = ["#1E90FF", "#FFD700", "#ff1493"];

export default class MapVision extends Component {
  constructor(props) {
    super(props);
    this.state = {
      pieOption: {},
      noise_number: 0,
      no_cancer_number: 0,
      cancer_number: 0,
      high_cancer_number: 0,
      total_num: 0,
      path: null,
      path2: null,
      heatEvent: null,
      heatMapType: "close",
      noiseChildren: [],
      noiseChildren2: [],
      index: [],
      index2: [],
      file_name: [],
      choosePatches: props.choosePatches,
      visble2: false,
      load: true,
      visible: false,
      gridSize: 1,
      confirmLoading: false,
      submitLoading: false,
      barOption: {},
      areaOption: {},
      selectedPatch: {},
      tipShowColor: [],
      high_cancer_rate: 0,
      cancer_rate: 0,
      no_cancer_rate: 0,
      drawerVisible: false,
      data_epo: 0,
      barOption2: {},
      barOption3: {},
      do_data: [],
      isModalVisible: false,
      isModalVisible2: false,
      isModalVisible3: false,
      total_num_2: 0,
      bmm_score: 0,
      spr_score: 0,
      value1: null,
      value2: null,
      judge:0,
      count:0,
      data_total:0
    };
  }
  check = () => {
    this.setState({ isModalVisible2: true })
  }
  newFunction = () => {
    const imgId = this.state.choosePatches[this.state.choosePatches.length - 1];
    var This = this;
    console.log("KKKKKKKKKK" + imgId)
    var do_data = [];
    var data_len = Object.keys(this.props.sample_Data.class).length;
    console.log("LLLLLL" + data_len)
    for (var i = 0; i < data_len; i++) {
      if (This.props.sample_Data.img_id[i] == imgId && This.props.sample_Data.is_labeled[i] == 0) {
        console.log(This.props.sample_Data.img_id[i] + "LLLLLLLL" + imgId)
        do_data.push([
          This.props.sample_Data.file_name[i].split("_").slice(0, 4).join("_") + ".png"
        ]);
      }
    }
    console.log("this is the dodate" + do_data)
    this.setState({ do_data: do_data });
  };
  noiseFilter = (value) => {
    this.setState({
      path: this.props.sample_Data.patch_id[value],
      path2: this.props.sample_Data.img_id[value]
    });
    var x_y = this.props.sample_Data.file_name[value].split("_");

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 这里数据切割问题,y和x是反过来的
    var y = parseInt(x_y[1]);
    var x = parseInt(x_y[3].split("png")[0]);
    this.setIndex(x, y);
    this.setVisible(true, value);
  };
  close = () => {
    this.setState({
      visble2: false
    });
  }
  noiseFilter2 = (value) => {
    var get = this.props.sample_Data.CAM_file_name[value].split(".");
    console.log(get)
    var target = parseInt(get[0]);
    this.setIndex2(target);
  };
  collect = () => {
    var mid = [];
    for (var i = 0; i <= 5000; i++) {
      mid.push(this.props.sample_Data.file_name[i]);
    }
    return mid;
  }
  collect_again = () => {
    var mid2 = [];
    for (var i = 0; i <= 5000; i++) {
      mid2.push(this.props.sample_Data.img_id[i]);
    }
    return mid2;
  }
  closeMap = () => {
    this.props.closeMap();
  };
  openDraw = () => {
    this.setState({ visible: true });
  };
  closeDraw = () => {
    this.setState({ drawerVisible: false });
  };
  setIndex = (x, y) => {
    this.setState({
      index: [x, y],
    });
  };
  setIndex2 = (h) => {
    this.setState({
      index2: [h],
    });
  };
  setVisible = async (args, key) => {
    var iter = [];
    for (var i = 1; i <= this.props.current_iteration; i++) iter.push(i);
    var bmms = this.props.sample_Data.bmm_num[key].slice(1, -1).split(",");
    var sprs = this.props.sample_Data.spr_num[key].slice(1, -1).split(",");
    bmms.forEach(function (obj) {
      obj = parseFloat(obj);
    });
    sprs.forEach(function (obj) {
      obj = parseFloat(obj);
    });
    await this.setState({
      barOption: {
        grid: {
          left: "30%",
          right: "10%",
        },
        yAxis: {
          type: "category",
          data: ["Bmm", "Spr"],
        },
        xAxis: {
          type: "value",
          min: 0.0,
          max: 1.0,
          show: false,
          splitLine: {
            //网格线
            show: false,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        series: [
          {
            data: [{
              value: this.props.sample_Data.bmm[key],
              itemStyle: {
                color: "#5470c6"
              }
            },
            {
              value: this.props.sample_Data.spr[key],
              itemStyle: {
                color: "#93cd77"
              }
            }


            ],
            type: "bar",
          },
        ],
      },
      areaOption: {
        grid: {
          top: 0,
          left: "13%",
          right: "13%",
          bottom: 20,
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "cross",
            animation: false,
            label: {
              backgroundColor: "#505765",
            },
          },
        },
        xAxis: [
          {
            type: "category",
            boundaryGap: false,
            axisLine: { onZero: false },
            // prettier-ignore
            data: iter,
          },
        ],
        yAxis: [
          {
            name: "Bmm",
            type: "value",
            min: 0,
            max: 1,
          },
          {
            name: "Spr",
            nameLocation: "start",
            alignTicks: true,
            type: "value",
            inverse: true,
            min: 0,
            max: 1,
          },
        ],
        series: [
          {
            name: "Bmm",
            type: "line",
            areaStyle: {},
            lineStyle: {
              width: 1,
            },
            showSymbol: false,
            emphasis: {
              focus: "series",
            },
            markArea: {
              silent: true,
              itemStyle: {
                opacity: 0.3,
              },
            },
            data: bmms,
          },
          {
            name: "Spr",
            type: "line",
            yAxisIndex: 1,
            showSymbol: false,
            areaStyle: {},
            lineStyle: {
              width: 1,
            },
            emphasis: {
              focus: "series",
            },
            markArea: {
              silent: true,
              itemStyle: {
                opacity: 0.3,
              },
            },
            data: sprs,
          },
        ],
      },
    });
    this.setState({
      visible: args,
    });
  };
  setVisible2 = async (valuea, valueb) => {
    await this.setState({
      barOption2: {
        grid: {
          left: "30%",
          right: "10%",
        },
        yAxis: {
          type: "category",
          data: ["本轮", "上一轮"],
        },
        xAxis: {
          type: "value",
          min: 400.0,
          max: 600.0,
          show: false,
          splitLine: {
            //网格线
            show: false,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        series: [
          {
            data: [{
              value: valuea,
              itemStyle: {
                color: "#5470c6"
              }
            },
            {
              value: valueb,
              itemStyle: {
                color: "#93cd77"
              }
            }
            ],
            type: "bar",
          },
        ],
      },
    });
  };
  setVisible3 = async (valuea, valueb) => {
    await this.setState({
      barOption3: {
        grid: {
          left: "30%",
          right: "10%",
        },
        yAxis: {
          type: "category",
          data: ["本轮", "上一轮"],
        },
        xAxis: {
          type: "value",
          min: 0.0,
          max: 100.0,
          show: false,
          splitLine: {
            //网格线
            show: false,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        series: [
          {
            data: [{
              value: valuea,
              itemStyle: {
                color: "#5470c6"
              }
            },
            {
              value: valueb,
              itemStyle: {
                color: "#93cd77"
              }
            }
            ],
            type: "bar",
          },
        ],
      },
    });
  };
  setPieChart = async (value1, value2, value3) => {
    const pieData = [
      { name: "无癌症", value: value1, itemStyle: { color: '#40b373' } }, // 绿色
      { name: "有癌症", value: value2, itemStyle: { color: '#d4b446' } }, // 蓝色
      { name: "高癌症", value: value3, itemStyle: { color: '#ee8826' } }, // 红色
    ];

    await this.setState({
      pieOption: {
        tooltip: {
          trigger: "item",
          formatter: "{a} <br/>{b}: {c} ({d}%)",
        },
        legend: {
          orient: "vertical",
          left: 10,
          data: ["无癌症", "有癌症", "高癌症"], // 更新图例数据
        },
        series: [
          {
            name: "信息",
            type: "pie",
            radius: ["50%", "70%"],
            avoidLabelOverlap: false,
            label: {
              show: false,
              position: "center",
            },
            emphasis: {
              label: {
                show: true,
                fontSize: "30",
                fontWeight: "bold",
              },
            },
            labelLine: {
              show: false,
            },
            data: pieData,
          },
        ],
      },
    });
  };

  changeGridSize = async (e) => {
    this.setState({
      gridSize: e.target.value,
    });
    this.drawChart();
    console.log(this.state.gridSize)
  };

  setConfirmLoading = (args) => {
    this.setState({
      confirmLoading: args,
    });
  };
  check_new_result = () => {
    var T = this;
    this.setState({
      alert: {
        message: "loading!",
      },
      init_loading: 0,
    });
    // 发送 POST 请求
    axios
      .get("http://127.0.0.1:5000/change")
      .then(function (response) {
        console.log("load")
        console.log(response)
        if (response.data.load_status == 200) {
          T.setState({
            init_loading: 1,
            epoch_Data: response.data.epoch_Data,
            sample_Data: response.data.sample_data,
            WSI_Data: response.data.WSI_Data,
            confusion_Data: response.data.confusion_Data,
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
  }
  handleOk = () => {
    var T = this;
    this.setConfirmLoading(true);
    setTimeout(() => {
      this.setVisible(false, 0);
      this.setConfirmLoading(false);
    }, 2000);
    this.check_new_result();

  };
  checkdetails = () => {
    this.setState({
      visble2: true
    });
  }
  handleSelectChange_2 = (value) => {
    const selectedLabel = category_name[value];
    console.log("Selected Label: ++++++++++++++", selectedLabel);
    console.log("patch_id", this.state.path)
    console.log("img_id", this.state.path2)
    axios
      .post('http://127.0.0.1:5000/last', {
        selectedLabel: selectedLabel,
        path: this.state.path,
        path2: this.state.path2
      })
      .then(response => {
        console.log('Response from server:', response.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    alert("修改成功！");
  }
  handleCancel = () => {
    this.setState({
      visible: false,
      isModalVisible2: false
    });
  };

  submitForTrain = () => {
    var This = this;
    this.setState({
      submitLoading: true,
    });

    message.loading(
      "Next AL-Iteration is running in progress. Please wait for minutes...",
      () => {
        axios
          .post("http://127.0.0.1:5000/train", {
            iteration: This.props.current_iteration,
          })
          .then(function (response) {
            // update
            This.setState({
              submitLoading: false,
            });
            message.success("New dataset of model has been updated!", 2.5);
          });
      }
    );
  };
  componentDidMount = async () => {
    await this.props.onChildEvent(this);
    this.drawChart();
    this.setState({
      load: false,
    });
  };

  changeHeat = (e) => {
    this.setState({
      heatMapType: e.target.value,
    });
    var c = [];
    if (e.target.value == "kmeans_label") {
      for (var i = 0; i < group_color.length; i++) {
        c.push([i != group_color.length - 1 ? "group-" + i : "background", group_color[i]]);
      }
    } else if (e.target.value == "category") {
      for (i = 0; i < category_color.length; i++) {
        c.push([category_name[i], category_color[i]]);
      }
    } else if (
      e.target.value == "bmm" ||
      e.target.value == "spr" ||
      e.target.value == "close"
    ) {
      for (i = 0; i < mapColors.length; i++) {
        c.push([mapLevel[i], mapColors[i]]);
      }
    }
    this.setState({
      tipShowColor: c,
    });
  };
  changeChoosePatches = async (p) => {
    await this.setState({
      choosePatches: p,
    });
    this.drawChart();

  };
  deleteImg = async (index) => {
    var tags = this.state.choosePatches;
    tags = tags.filter((item) => item != index);

    await this.setState({
      choosePatches: tags,
    });
    this.props.changeDeletePatches(tags);
  };
  chooseImg = async (img_id) => {
    this.props.showMap(img_id);
  };

  drawChart = async () => {
    this.newFunction();
    var data_len_epo = Object.keys(this.props.epoch_Data.acc).length;
    const colors = ["#e7f1ff", "#0c3b80"];
    const imgId = this.state.choosePatches[this.state.choosePatches.length - 1];
    const This = this;
    xScale.domain([0, rows]).range([0, imgSize * rows]);
    yScale.domain([0, cols]).range([imgSize * cols, 0]);
    d3.select("#map").selectAll("svg").remove();

    // 初始化zoom
    const zoom = d3
      .zoom()
      .scaleExtent([1, maxValue])
      .translateExtent([
        [-5, -5],
        [imgSize * rows, imgSize * cols],
      ])
      .on("zoom", zoomed);
    // 热力图
    var colorScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range(colors)
      .interpolate(d3.interpolateHcl);
    // 初始化画布
    const mainGroup = d3
      .select("#map")
      .append("svg")
      .attr("preserveAspectRatio", "xMinYMin meet")
      .attr("width", "100%")
      .attr("height", "100%");
    //导入数据
    var imgData = [];
    var bk_imgData = [];
    var file_name_collect = [];
    var noiseChild = [];
    var noiseChild2 = [];
    var backLocation = new Array(rows).fill(0).map(v => new Array(cols).fill(0));
    var backLocationIndex = [];
    var noise_number = 0;
    var no_cancer_number = 0;
    var cancer_number = 0;
    var high_cancer_number = 0;
    var data_len = Object.keys(this.props.sample_Data.class).length;
    var bk_data_len = Object.keys(this.props.bk_data.class).length;
    var total_num_2 = 0;
    var count = 0;
    this.setState({
      data_epo: data_len_epo
    })
    for (var i = 0; i < data_len; i++) {
      if (this.props.sample_Data.img_id[i] == imgId) {
        total_num_2 = total_num_2 + 1;
        file_name_collect.push(<Option key={i}>{this.props.sample_Data.file_name[i]}</Option>);
        var x_y = this.props.sample_Data.file_name[i].split("_");
        var y = parseInt(x_y[1]);
        var x = parseInt(x_y[3].split("png")[0]);
        backLocation[y][x] = 1;
        imgData.push({
          patch_id: this.props.sample_Data.patch_id[i],
          bmm: this.props.sample_Data.bmm[i],
          spr: this.props.sample_Data.spr[i],
          heat_score: this.props.sample_Data.heat_score[i],
          file_name: this.props.sample_Data.file_name[i],
          noise: this.props.sample_Data.noise[i],
          class: this.props.sample_Data.class[i],
          //new
          bmm_num: this.props.sample_Data.bmm_num[i],
          spr_num: this.props.sample_Data.spr_num[i],
          CAM_file: this.props.sample_Data.CAM_file_name[i],
          is_labeled: this.props.sample_Data.is_labeled[i],
          kmeans_label: this.props.sample_Data.kmeans_label[i],
        });
        if (this.props.sample_Data.noise[i] == 1 && this.props.sample_Data.is_labeled[i] == 1 ) {
          noiseChild.push(
            <Option key={i}>{this.props.sample_Data.file_name[i]}</Option>
          );
          noiseChild2.push(
            <Option key={i}>{this.props.sample_Data.CAM_file_name[i]}</Option>
          );
          noise_number = noise_number + 1;
        }
        if (this.props.sample_Data.class[i] == 1) {
          no_cancer_number = no_cancer_number + 1;
        }
        if (this.props.sample_Data.class[i] == 2) {
          cancer_number = cancer_number + 1;
        }
        if (this.props.sample_Data.class[i] == 3) {
          high_cancer_number = high_cancer_number + 1;
        }
      }
      if(this.props.sample_Data.is_labeled[i] == 0){
        count += 1;
      }
      var high_cancer_rate;
      var cancer_rate;
      var no_cancer_rate;
      var judge = 0;
     

      this.setPieChart(no_cancer_number, cancer_number, high_cancer_number);
    }
    high_cancer_rate = high_cancer_number / total_num_2;
    cancer_rate = cancer_number / total_num_2;
    no_cancer_rate = no_cancer_number / total_num_2;
    console.log("high_cancer_rate" + high_cancer_rate );
    console.log("cancer_rate" + cancer_rate )
    console.log("no_cancer_rate" + no_cancer_rate )
    if(high_cancer_rate > cancer_rate && high_cancer_rate > no_cancer_rate &&high_cancer_rate < 0.2 && cancer_rate < 0.3 ){
      console.log("we have entered here ! ")
      judge = 1;
   }
   if(cancer_rate > high_cancer_rate && cancer_rate > no_cancer_rate  && high_cancer_rate < 0.2 && cancer_rate < 0.3){
      judge = 2;
   }
   if(no_cancer_rate > high_cancer_rate && no_cancer_rate > cancer_rate  ){
    console.log(this.state.high_cancer_rate + "LLLLLLLLL")
    if(high_cancer_rate > 0.2){
      judge = 4;
    }
    if(cancer_rate > 0.3){
      judge = 5;
    }
    else {
      judge = 3;
    } 
   }
    this.setState({
      count :count,
      data_total : data_len,
      judge:judge,
      noise_number: noise_number ,
      no_cancer_number: no_cancer_number,
      cancer_number: cancer_number,
      high_cancer_number: high_cancer_number,
      total_num: noise_number + no_cancer_number + cancer_number + high_cancer_number,
      high_cancer_rate: high_cancer_rate,
      cancer_rate: cancer_rate,
      no_cancer_rate: no_cancer_rate
    });
    this.setState({ total_num_2: total_num_2 });
    console.log("this is the total num : " + total_num_2)
    for (i = 0; i < bk_data_len; i++) {
      if (this.props.bk_data.img_id[i] == imgId) {
        var x_y = this.props.bk_data.file_name[i].split("_");
        var y = parseInt(x_y[1]);
        var x = parseInt(x_y[3].split("png")[0]);
        backLocation[y][x] = 1;
        bk_imgData.push({
          patch_id: this.props.bk_data.patch_id[i],
          file_name: this.props.bk_data.file_name[i],
          class: this.props.bk_data.class[i],
          kmeans_label: this.props.bk_data.kmeans_label[i],
        })
      }
    }
    // 格式化
    var bmm_range = [2, -1]
    var spr_range = [2, -1]
    var smooth = 1e-5
    for (var i = 0; i < imgData.length; i++) {
      bmm_range[0] = imgData[i]['bmm'] < bmm_range[0] ? imgData[i]['bmm'] : bmm_range[0]
      bmm_range[1] = imgData[i]['bmm'] > bmm_range[1] ? imgData[i]['bmm'] : bmm_range[1]
      spr_range[0] = imgData[i]['spr'] < spr_range[0] ? imgData[i]['spr'] : spr_range[0]
      spr_range[1] = imgData[i]['spr'] > spr_range[1] ? imgData[i]['spr'] : spr_range[1]
    }
    for (var i = 0; i < imgData.length; i++) {
      imgData[i]['bmm'] = (imgData[i]['bmm'] - bmm_range[0]) / (bmm_range[1] - bmm_range[0] + smooth)
      imgData[i]['spr'] = (imgData[i]['spr'] - spr_range[0]) / (spr_range[1] - spr_range[0] + smooth)
    }
    // 找边界
    var min_y = 100;
    var max_y = 0;
    for (var i = 0; i < rows; i++)
      for (var j = 0; j < cols; j++) {
        if (backLocation[i][j] == 1) {
          min_y = Math.min(i, min_y)
          max_y = Math.max(i, max_y)
        }
      }

    for (i = min_y; i <= max_y; i++) {
      var min_x = 100;
      var max_x = 0;
      for (j = 0; j < cols; j++)
        if (backLocation[i][j] == 1) {
          min_x = Math.min(j, min_x)
          max_x = Math.max(j, max_x)
        }
      for (j = min_x; j <= max_x; j++) {
        if (backLocation[i][j] == 0)
          backLocationIndex.push([i, j]);
      }
    }
    this.setState({
      noiseChildren: noiseChild,
      noiseChildren2: noiseChild2,
    });

    // 添加
    mainGroup.call(zoom);
    drawGrid();
    drawPatches();
    // 恢复大小
    d3.select("#zoom_out").on("click", () => {
      mainGroup.transition().call(zoom.transform, d3.zoomIdentity, [0, 0]);
      mainGroup.selectAll("rect").remove();
      This.setState({
        heatMapType: "close"
      })
    }

    );
    // 改变热力图
    d3.select("#zoom_change").on("change", toHeatMap);
    d3.select("#zoom_change1").on("change", toHeatMap);


    // 绘制网格
    function drawGrid(event) {
      mainGroup.selectAll("line").remove();
      var margin = lineWidth;
      if (event != null && event.transform.k > 25)
        margin = (lineWidth * event.transform.k) / 20;
      var grid = (g) =>
        g
          .attr("stroke", "blue")
          .attr("stroke-opacity", 0.5)
          .attr("stroke-width", margin)
          .call((g) =>
            g
              .append("g")
              .selectAll("line")
              .data(xScale.ticks(rows))
              .join("line")
              .attr("x1", (d) => xScale(This.state.gridSize * d))
              .attr("x2", (d) => xScale(This.state.gridSize * d))
              .attr("y2", imgSize * cols)
              .attr("transform", event == null ? null : event.transform)
          )
          .call((g) =>
            g
              .append("g")
              .selectAll("line")
              .data(yScale.ticks(cols))
              .join("line")
              .attr("y1", (d) => yScale(This.state.gridSize * d))
              .attr("y2", (d) => yScale(This.state.gridSize * d))
              .attr("x2", imgSize * rows)
              .attr("transform", event == null ? null : event.transform)
          );
      mainGroup.call(grid);
    }
    // 绘图
    function drawPatches(event) {
      var p = process.env.REACT_APP_IMAGE_PATH + "/image_" + imgId;
      console.log("this is the turth" + imgId);
      console.log(p)
      mainGroup.selectAll("image").remove();
      var margin = lineWidth;
      if (event != null && event.transform.k > 25)
        margin = (lineWidth * event.transform.k) / 20;

      const imgs = mainGroup.selectAll("image").data([0]);

      imgData.forEach((img, key) => {
        var x_y = img["file_name"].split("_");
        var arrayLength = x_y.length;
        var y = parseInt(x_y[1]);

        var x = parseInt(x_y[3].split("png")[0]);
        if (arrayLength == 5) {
          var path =
            process.env.REACT_APP_IMAGE_PATH +
            "/image_" +
            imgId +
            "/" +
            // img["file_name"].split("_").slice(0, 4).join("_") + ".png";
            img['file_name'];
        }
        else {
          var path =
            process.env.REACT_APP_IMAGE_PATH_NEW +
            "/image_" +
            imgId +
            "/" +
            img["file_name"].split("_").slice(0, 4).join("_") + ".png";
          // img['file_name'];
        }
        // var path =
        //   process.env.REACT_APP_IMAGE_PATH + 
        //   "/image_" +
        //   imgId +
        //   "/" +
        //   // img["file_name"].split("_").slice(0, 4).join("_") + ".png";
        //   img['file_name'];
        imgs
          .enter()
          .append("svg:image")
          .attr("xlink:href", path)
          .attr("row", x)
          .attr("col", y)
          .attr("x", imgSize * x + margin)
          .attr("y", imgSize * y + margin)
          .attr("img_id", imgId)
          .attr("patch_id", img["patch_id"])
          .attr("width", imgSize - lineWidth - margin)
          .on("mouseover", function (d) {
            d3.select(this)
              .attr("width", imgSize * 1.2)
              .attr("height", imgSize * 1.2);
          })
          .on("mouseout", function (d) {
            d3.select(this)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin);
          })
          .on("click", async function (d) {
            const row = this.getAttribute("row");
            const col = this.getAttribute("col");

            await This.setIndex(row, col);
            await This.setState({
              selectedPatch: key,
            });
            This.setVisible(true, key);
          })
          .attr("transform", event == null ? null : event.transform);
      });
      bk_imgData.forEach((img, key) => {
        var x_y = img["file_name"].split("_");
        var arrayLength = x_y.length;
        var y = parseInt(x_y[1]);

        var x = parseInt(x_y[3].split("png")[0]);
        if (arrayLength == 5) {
          var path =
            process.env.REACT_APP_IMAGE_PATH +
            "/image_" +
            imgId +
            "/" +
            // img["file_name"].split("_").slice(0, 4).join("_") + ".png";
            img['file_name'];
        }
        else {
          var path =
            process.env.REACT_APP_IMAGE_PATH_NEW +
            "/image_" +
            imgId +
            "/" +
            img["file_name"].split("_").slice(0, 4).join("_") + ".png";
          // img['file_name'];
        }
        imgs
          .enter()
          .append("svg:image")
          .attr("xlink:href", path)
          .attr("row", x)
          .attr("col", y)
          .attr("x", imgSize * x + margin)
          .attr("y", imgSize * y + margin)
          .attr("img_id", imgId)
          .attr("patch_id", img["patch_id"])
          .attr("width", imgSize - lineWidth - margin)
          .on("mouseover", function (d) {
            d3.select(this)
              .attr("width", imgSize * 1.2)
              .attr("height", imgSize * 1.2);
          })
          .on("mouseout", function (d) {
            d3.select(this)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin);
          })
          .attr("transform", event == null ? null : event.transform);
      })
    }

    //绘制热力图
    async function drawHeatMap() {
      if (This.state.heatMapType != "close") {
        mainGroup.selectAll("rect").remove();
        try {
          var margin = lineWidth / This.state.heatEvent.transform.k;
        } catch (err) {
          var margin = lineWidth;
        }
        setTimeout(() => {
          mainGroup.selectAll("rect").remove();

          imgData.forEach((img) => {
            var x_y = img["file_name"].split("_");
            var length = x_y.length
            //x_11_y_12_0.png
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 这里数据切割问题,y和x是反过来的
            var y = parseInt(x_y[1]);
            var x = parseInt(x_y[3].split("png")[0]);
            if (length == 5) {
              var lb = parseInt(x_y[4].split(".")[0]);
            }
            else {
              var lb = parseInt(x_y[5].split(".")[0]);
            }
            if (y == 8 && x == 2) {
              console.log("+++++++++++++" + lb)
            }
            mainGroup
              .append("g")
              .append("rect") //添加类型
              .attr("x", imgSize * x + margin)
              .attr("y", imgSize * y + margin)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin)
              .attr("fill", function () {
                if (This.state.heatMapType == "category")
                  return category_color[lb - 1];
                else if (This.state.heatMapType == "kmeans_label")
                  return group_color[parseInt(img[This.state.heatMapType])];
                else if (This.state.heatMapType == "bmm")
                  return colorScale(parseFloat(img[This.state.heatMapType] + 0.2));
                else return colorScale(parseFloat(img[This.state.heatMapType]));
              })
              .attr("opacity", 0.8)
              .attr(
                "transform",
                This.state.heatEvent == null
                  ? null
                  : This.state.heatEvent.transform
              );
          });
          bk_imgData.forEach((img) => {
            var x_y = img["file_name"].split("_");
            var length = x_y.length
            var y = parseInt(x_y[1]);
            var x = parseInt(x_y[3].split("png")[0]);
            if (length == 5) {
              var lb = parseInt(x_y[4].split(".")[0]);
            }
            else {
              var lb = parseInt(x_y[5].split(".")[0]);
            }
            if (y == 8 && x == 2) {
              console.log("+++++++++++++" + lb)
            }
            mainGroup
              .append("g")
              .append("rect") //添加类型
              .attr("x", imgSize * x + margin)
              .attr("y", imgSize * y + margin)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin)
              .attr("fill", function () {
                if (This.state.heatMapType == "category")
                  return category_color[lb - 1];
                else if (This.state.heatMapType == "kmeans_label")
                  return group_color[parseInt(img[This.state.heatMapType])];
                else if (This.state.heatMapType == "kmeans_label")
                  return group_color[parseInt(img[This.state.heatMapType])];
                else if (This.state.heatMapType == "bmm")
                  return colorScale(0.1);
                else return colorScale(0.8);
              })
              .attr("opacity", 0.8)
              .attr(
                "transform",
                This.state.heatEvent == null
                  ? null
                  : This.state.heatEvent.transform
              );
          });
          backLocationIndex.forEach((item) => {
            mainGroup
              .append("g")
              .append("rect") //添加类型
              .attr("x", imgSize * item[1] + margin)
              .attr("y", imgSize * item[0] + margin)
              .attr("width", imgSize - margin)
              .attr("height", imgSize - margin)
              .attr("fill", function () {
                if (This.state.heatMapType == "category")
                  return category_color[0];
                else if (This.state.heatMapType == "bmm")
                  return colorScale(0.1);
                else return colorScale(0.8);
              })
              .attr("opacity", 0.8)
              .attr(
                "transform",
                This.state.heatEvent == null
                  ? null
                  : This.state.heatEvent.transform
              );
          })
        }, 0);
      }
    }

    async function toHeatMap() {
      var type = This.state.heatMapType;

      if (type != "close") {
        await drawHeatMap();
      } else {
        mainGroup.selectAll("rect").remove();
      }
    }
    function zoomed(event) {
      This.setState({
        heatEvent: event,
      });
      drawGrid(event);
      drawPatches(event);
      drawHeatMap(event);
    }
    this.setVisible2(this.props.epoch_Data.labeled[data_len_epo - 1], this.props.epoch_Data.labeled[data_len_epo - 2]);
    this.setVisible3(this.props.epoch_Data.unlabeled[data_len_epo - 1], this.props.epoch_Data.unlabeled[data_len_epo - 2]);

  };

  dowork = (value1) => {
    this.setState({ isModalVisible3: true });
    const This = this;
    const imgId = this.state.choosePatches[this.state.choosePatches.length - 1];
    var data_len = Object.keys(this.props.sample_Data.class).length;
    var bmm_score = 0;
    var spr_score = 0;
    var value2 = null;
    for (var i = 0; i < data_len; i++) {
      if (This.props.sample_Data.img_id[i] == imgId && This.props.sample_Data.file_name[i].split("_").slice(0, 4).join("_") + ".png" == value1) {
        bmm_score = This.props.sample_Data.bmm[i];
        spr_score = This.props.sample_Data.spr[i];
        value2 = This.props.sample_Data.file_name[i];
      }
    }
    this.setState({ bmm_score: bmm_score, spr_score: spr_score, value1: value1, value2: value2 })
  }
  click1 = () => {
    console.log("img_id", this.state.choosePatches[this.state.choosePatches.length - 1])
    console.log("patch_id", this.state.value2)
    axios
      .post('http://127.0.0.1:5000/last2', {
        img_id: this.state.choosePatches[this.state.choosePatches.length - 1],
        patch_id: this.state.value2
      })
      .then(response => {
        console.log('Response from server:', response.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    alert("您已经成功标记这个切片!");
    this.setState({ isModalVisible3: false });
    window.location.reload();
  }
  click2 = () => {
    console.log("img_id", this.state.choosePatches[this.state.choosePatches.length - 1])
    console.log("patch_id", this.state.value2)
    axios
      .post('http://127.0.0.1:5000/last3', {
        img_id: this.state.choosePatches[this.state.choosePatches.length - 1],
        patch_id: this.state.value2
      })
      .then(response => {
        console.log('Response from server:', response.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    alert("您已经成功标记这个切片!");
    this.setState({ isModalVisible3: false });
    window.location.reload();
  }
  click3 = () => {
    console.log("img_id", this.state.choosePatches[this.state.choosePatches.length - 1])
    console.log("patch_id", this.state.value2)
    axios
      .post('http://127.0.0.1:5000/last4', {
        img_id: this.state.choosePatches[this.state.choosePatches.length - 1],
        patch_id: this.state.value2
      })
      .then(response => {
        console.log('Response from server:', response.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    alert("您已经成功标记这个切片!");
    this.setState({ isModalVisible3: false });
    window.location.reload();
  }
  render() {
    return (
      <>

        <Spin id="loading" size="large" spinning={this.state.load}>
          <Card bordered={false} hoverable={true}>
            <Row align="center" gutter={[5, 3]}>
              <Col span={24} style={{ height: '60px' }} >
                <Row gutter={10}>
                  <Col span={3}>
                    <Title level={5}>WSI-{this.state.choosePatches[this.state.choosePatches.length - 1] + 1}</Title>
                  </Col>
                  <Col span={3}>
                    <Text type="secondary">
                      <AimOutlined />
                      &nbsp;刷新:
                    </Text>
                  </Col>
                  <Col span={4}>
                    <Tooltip
                      placement="top"
                      title={"Click for initial position"}
                    >
                      <Button
                        id="zoom_out"
                        type="primary"
                        shape="round"
                        size="small"
                        icon={<RedoOutlined />}
                        ghost
                      >
                        刷新
                      </Button>
                    </Tooltip>
                  </Col>
                  <Col offset={4} span={4}>
                    <Text type="secondary">
                      <MenuOutlined />
                      &nbsp;查看噪声列表:
                    </Text>
                  </Col>
                  <Col span={4}>
                    <Button size="small" onClick={this.openDraw} shape="round">
                      噪声列表
                    </Button>

                    <Modal
                      title="Noise List"
                      visible={this.state.visible}
                      onOk={this.handleOk}
                      confirmLoading={this.state.confirmLoading}
                      onCancel={this.handleCancel}
                      width={800}

                    >
                      <Col style={{ marginBottom: '10px' }}>
                        <Text style={{ marginTop: '10px' }}>
                          WSI number : WSI-{this.state.choosePatches[this.state.choosePatches.length - 1] + 1}
                        </Text>
                      </Col>
                      <Row style={{ marginBottom: '30px' }}>
                        <Row>
                          <Text style={{ marginRight: '10px' }}>
                            Choose noise patches:
                          </Text>
                          <Select
                            allowClear
                            size="small"
                            placeholder="No data"
                            onChange={this.noiseFilter}
                          >
                            {this.state.noiseChildren}
                          </Select>
                        </Row>
                        <Row style={{ marginLeft: '200px' }} >
                          <Text style={{ marginRight: '10px' }}>
                            Choose CAM files:
                          </Text>
                          <Select
                            allowClear
                            size="small"
                            // style={{ width: "100%" }}
                            placeholder="No data"
                            onChange={this.noiseFilter2}
                          >
                            {this.state.noiseChildren2}
                          </Select>
                        </Row>
                      </Row>
                      <Row gutter={[0, 10]} justify="space-around" align="middle">
                        <Col span={12} offset={1}>
                          <ShowImg
                            index={this.state.index}
                            imgId={this.state.choosePatches[this.state.choosePatches.length - 1] + 1}
                            file_name={this.collect()}
                            id_name={this.collect_again()}
                          />
                        </Col>
                        <Col span={10} offset={1}>
                          <Text type="secondary">Grad-Cam Image</Text>
                          <Image
                            fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3PTWBSGcbGzM6GCKqlIBRV0dHRJFarQ0eUT8LH4BnRU0NHR0UEFVdIlFRV7TzRksomPY8uykTk/zewQfKw/9znv4yvJynLv4uLiV2dBoDiBf4qP3/ARuCRABEFAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghggQAQZQKAnYEaQBAQaASKIAQJEkAEEegJmBElAoBEgghgg0Aj8i0JO4OzsrPv69Wv+hi2qPHr0qNvf39+iI97soRIh4f3z58/u7du3SXX7Xt7Z2enevHmzfQe+oSN2apSAPj09TSrb+XKI/f379+08+A0cNRE2ANkupk+ACNPvkSPcAAEibACyXUyfABGm3yNHuAECRNgAZLuYPgEirKlHu7u7XdyytGwHAd8jjNyng4OD7vnz51dbPT8/7z58+NB9+/bt6jU/TI+AGWHEnrx48eJ/EsSmHzx40L18+fLyzxF3ZVMjEyDCiEDjMYZZS5wiPXnyZFbJaxMhQIQRGzHvWR7XCyOCXsOmiDAi1HmPMMQjDpbpEiDCiL358eNHurW/5SnWdIBbXiDCiA38/Pnzrce2YyZ4//59F3ePLNMl4PbpiL2J0L979+7yDtHDhw8vtzzvdGnEXdvUigSIsCLAWavHp/+qM0BcXMd/q25n1vF57TYBp0a3mUzilePj4+7k5KSLb6gt6ydAhPUzXnoPR0dHl79WGTNCfBnn1uvSCJdegQhLI1vvCk+fPu2ePXt2tZOYEV6/fn31dz+shwAR1sP1cqvLntbEN9MxA9xcYjsxS1jWR4AIa2Ibzx0tc44fYX/16lV6NDFLXH+YL32jwiACRBiEbf5KcXoTIsQSpzXx4N28Ja4BQoK7rgXiydbHjx/P25TaQAJEGAguWy0+2Q8PD6/Ki4R8EVl+bzBOnZY95fq9rj9zAkTI2SxdidBHqG9+skdw43borCXO/ZcJdraPWdv22uIEiLA4q7nvvCug8WTqzQveOH26fodo7g6uFe/a17W3+nFBAkRYENRdb1vkkz1CH9cPsVy/jrhr27PqMYvENYNlHAIesRiBYwRy0V+8iXP8+/fvX11Mr7L7ECueb/r48eMqm7FuI2BGWDEG8cm+7G3NEOfmdcTQw4h9/55lhm7DekRYKQPZF2ArbXTAyu4kDYB2YxUzwg0gi/41ztHnfQG26HbGel/crVrm7tNY+/1btkOEAZ2M05r4FB7r9GbAIdxaZYrHdOsgJ/wCEQY0J74TmOKnbxxT9n3FgGGWWsVdowHtjt9Nnvf7yQM2aZU/TIAIAxrw6dOnAWtZZcoEnBpNuTuObWMEiLAx1HY0ZQJEmHJ3HNvGCBBhY6jtaMoEiJB0Z29vL6ls58vxPcO8/zfrdo5qvKO+d3Fx8Wu8zf1dW4p/cPzLly/dtv9Ts/EbcvGAHhHyfBIhZ6NSiIBTo0LNNtScABFyNiqFCBChULMNNSdAhJyNSiECRCjUbEPNCRAhZ6NSiAARCjXbUHMCRMjZqBQiQIRCzTbUnAARcjYqhQgQoVCzDTUnQIScjUohAkQo1GxDzQkQIWejUogAEQo121BzAkTI2agUIkCEQs021JwAEXI2KoUIEKFQsw01J0CEnI1KIQJEKNRsQ80JECFno1KIABEKNdtQcwJEyNmoFCJAhELNNtScABFyNiqFCBChULMNNSdAhJyNSiECRCjUbEPNCRAhZ6NSiAARCjXbUHMCRMjZqBQiQIRCzTbUnAARcjYqhQgQoVCzDTUnQIScjUohAkQo1GxDzQkQIWejUogAEQo121BzAkTI2agUIkCEQs021JwAEXI2KoUIEKFQsw01J0CEnI1KIQJEKNRsQ80JECFno1KIABEKNdtQcwJEyNmoFCJAhELNNtScABFyNiqFCBChULMNNSdAhJyNSiECRCjUbEPNCRAhZ6NSiAARCjXbUHMCRMjZqBQiQIRCzTbUnAARcjYqhQgQoVCzDTUnQIScjUohAkQo1GxDzQkQIWejUogAEQo121BzAkTI2agUIkCEQs021JwAEXI2KoUIEKFQsw01J0CEnI1KIQJEKNRsQ80JECFno1KIABEKNdtQcwJEyNmoFCJAhELNNtScABFyNiqFCBChULMNNSdAhJyNSiEC/wGgKKC4YMA4TAAAAABJRU5ErkJggg=="
                            width={160}
                            height={160}
                            src={
                              "./data/init_data_image" +
                              "/image_CAM_" +
                              this.props.chooseMapImg +
                              "/" +
                              this.state.index2 +
                              ".png"
                            }

                          />
                        </Col>
                        <Col span={24}>
                          <Descriptions layout="vertical" size="large" column={4} bordered>
                            <Descriptions.Item span={2} label="Bmm & Spr Scores">
                              <ReactECharts
                                style={{ width: 190, height: 80 }}
                                option={this.state.barOption}
                              />
                            </Descriptions.Item>
                            <Descriptions.Item span={2} label="Trends">
                              <ReactECharts
                                style={{ width: 200, height: 110 }}
                                option={this.state.areaOption}
                              />
                            </Descriptions.Item>

                            <Descriptions.Item span={4} label="Update Class">
                              <Select
                                style={{ width: "100%" }}
                                placeholder="select one class"
                                optionLabelProp="label"
                                onChange={(value) => this.handleSelectChange_2(value)}
                              >
                                {category_name.map((item, index) => {
                                  return (
                                    <Option value={index} label={item}>
                                      <div className="demo-option-label-item">
                                        <Tag color={category_color[index]}>{item}</Tag>
                                      </div>
                                    </Option>
                                  );
                                })}
                              </Select>
                            </Descriptions.Item>
                          </Descriptions>
                        </Col>
                      </Row>

                    </Modal>
                  </Col>
                  <Col offset={3} span={2}>
                    <Text type="secondary">
                      <RadarChartOutlined />标签:
                    </Text>
                  </Col>
                  <Col span={4}>
                    <Radio.Group
                      id="zoom_change"
                      value={this.state.heatMapType}
                      onChange={this.changeHeat}
                      size="small"
                      buttonStyle="solid"
                    >
                      {/* <Radio.Button value="kmeans_label" selected>Group</Radio.Button> */}
                      <Radio.Button value="category">类别</Radio.Button>
                    </Radio.Group>
                  </Col>
                  <Col offset={5} span={3}>
                    <Text type="secondary">
                      <HeatMapOutlined />算法得分:
                    </Text>
                  </Col>
                  <Col span={5}>
                    <Radio.Group
                      id="zoom_change1"
                      value={this.state.heatMapType}
                      onChange={this.changeHeat}
                      size="small"
                      buttonStyle="solid"
                    >
                      <Radio.Button value="bmm" selected>Bmm算法</Radio.Button>
                      <Radio.Button value="spr">PR算法</Radio.Button>
                    </Radio.Group>
                  </Col>
                </Row>
              </Col>
              <Col span={24} style={{ height: '450px' }}>
                <div id="map" style={{ height: '450px' }}>
                  <div id="tooltipMap" >
                    <Row gutter={5}>
                      {this.state.tipShowColor.map((item, index) => {
                        return (
                          <>
                            <Col offset={2} span={4}>
                              <div
                                style={{
                                  width: 15,
                                  height: 15,
                                  backgroundColor: item[1],
                                }}
                              ></div>
                            </Col>
                            <Col span={18}>
                              <Text>{item[0]}</Text>
                            </Col>
                          </>
                        );
                      })}
                    </Row>
                  </div>
                </div>
              </Col>
              <Col span={24} >
                <Row gutter={[16, 0]}> {/* gutter 可以用来设置 Col 之间的间距 */}
                  <Col span={8}>
                    <div style={{ display: 'flex', flexDirection: 'column', padding: '16px', border: '1px solid #eee', borderRadius: '8px', boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text strong style={{ fontSize: '18px', color: '#333', marginBottom: '8px' }}>WSI-{this.state.choosePatches[this.state.choosePatches.length - 1] + 1}</Text> {/* 主文字 */}
                        <Button onClick={this.checkdetails} style={{ backgroundColor: '#1890ff', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer' }}>详情</Button>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '19px', color: '#666' }}>
                        <Text>噪声切片数量: {this.state.noise_number} </Text>
                        <Text>噪声率: <span style={{ fontWeight: 'bold' }}>{(this.state.noise_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                        <Text style={{ color: '#40b373' }}>无癌症概率切片数量: {this.state.no_cancer_number}</Text>
                        <Text style={{ color: '#40b373' }}>无癌症占比率: <span style={{ fontWeight: 'bold' }}>{(this.state.no_cancer_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                        <Text style={{ color: '#d4b446' }}>低癌症概率切片数量: {this.state.cancer_number}</Text>
                        <Text style={{ color: '#d4b446' }}>低癌症占比率: <span style={{ fontWeight: 'bold' }}>{(this.state.cancer_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                        <Text style={{ color: "#ee8826" }}>高癌症概率切片数量: {this.state.high_cancer_number}</Text>
                        <Text style={{ color: "#ee8826" }}>高癌症占比率: <span style={{ fontWeight: 'bold' }}>{(this.state.high_cancer_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={16}> {/* 右侧 Col，占比为 1 */}
                    <Row gutter={[0, 16]} style={{ display: 'flex', flexDirection: 'column' }}>
                      <Col span={24} style={{ flex: 1 }}>
                        <Typography.Title level={4} style={{ fontWeight: 'bold', marginBottom: '5px' }}>本切片标记的结果</Typography.Title>
                        <Descriptions column={2} bordered>
                          <Descriptions.Item label="标记切片数">{this.state.total_num_2 - this.state.do_data.length}</Descriptions.Item>
                          <Descriptions.Item label="未标记切片数">{this.state.do_data.length}</Descriptions.Item>

                        </Descriptions>
                      </Col>
                      <Col style={{ display: 'flex', alignItems: 'center', marginTop: '35px' }}>
                        <Text style={{ marginRight: '170px', marginLeft: '25px' }}>点击开始标记作业</Text>
                        <Button type="primary" onClick={this.check}>开始</Button>
                      </Col>
                      <Col span={24} style={{ flex: 1,marginTop:'35px' }}>
                        <Typography.Title level={4} style={{ fontWeight: 'bold', marginBottom: '5px' }}>切片工作量统计</Typography.Title>
                        <Descriptions column={2} bordered>
                          <Descriptions.Item label="标记切片数">{this.state.data_total - this.state.count}</Descriptions.Item>
                          <Descriptions.Item label="未标记切片数">{this.state.count}</Descriptions.Item>
                        </Descriptions>
                      </Col>
                      <Col span={24} style={{ flex: 1 }}>
                        <Text>在我们的模型下，在迭代后，已标注切片数量自动增加了{this.props.epoch_Data.labeled[this.state.data_epo - 1] -this.props.epoch_Data.labeled[this.state.data_epo - this.state.data_epo] }张</Text>
                      </Col>
                      {/* <Col span={24} style={{ flex: 1 }}>

                        <Row gutter={16}>
                          <Col span={12}>
                            <Typography.Title level={4} style={{ fontWeight: 'bold', marginBottom: '5px' }}>已标注样本变化趋势</Typography.Title>
                            <Descriptions.Item label="已标注样本">
                              <ReactECharts
                                style={{ width: '100%', height: '200px' }}
                                option={this.state.barOption2}
                              />
                            </Descriptions.Item>
                          </Col>
                          <Col span={12}>
                            <Typography.Title level={4} style={{ fontWeight: 'bold', marginBottom: '5px' }}>未标注样本变化趋势</Typography.Title>
                            <Descriptions.Item label="未标注样本">
                              <ReactECharts
                                style={{ width: '100%', height: '200px' }}
                                option={this.state.barOption3}
                              />
                            </Descriptions.Item>
                          </Col>
                        </Row>
                      </Col> */}
                    </Row>
                  </Col>
                </Row>
              </Col>
            </Row>
          </Card>
        </Spin>
        <Modal
          title="详情"
          visible={this.state.visble2}
          onCancel={this.close}
          centered
          footer={[
            <Button key="close" onClick={() => this.setState({ isModalVisible: false })}>关闭</Button>
          ]}
          width={1600}
        >
          <Row>
            <Col span={24}>
              <Descriptions layout="vertical" size="large" column={4} bordered>
                <Descriptions.Item span={2} label="WSI图片展示">
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Image width={300}
                      fallback=""
                      src={process.env.REACT_APP_WSI_PATH_NEW + "/" + this.state.choosePatches[this.state.choosePatches.length - 1] + ".png"}
                      alt={`Selected Image`}
                      style={{ maxWidth: '100%', height: 'auto' }}
                    />
                    <div style={{ marginLeft: '20px' }}>
                      <Text style={{ marginBottom: '5px', marginRight: '10px' }}>就诊号:20240422117</Text>
                      <Text style={{ marginBottom: '5px' }}>送检方式:病理切片</Text>
                    </div>
                  </div>
                </Descriptions.Item>
                <Descriptions.Item span={2} label="WSI图片预测结果数据" >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', color: '#666' }}>
                    <Text>噪声切片数量: {this.state.noise_number} </Text>
                    <Text>噪声率: <span style={{ fontWeight: 'bold' }}>{(this.state.noise_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                    <Text style={{ color: '#40b373' }}>无癌症概率切片数量: {this.state.no_cancer_number}</Text>
                    <Text style={{ color: '#40b373' }}>无癌症占比率: <span style={{ fontWeight: 'bold' }}>{(this.state.no_cancer_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                    <Text style={{ color: '#d4b446' }}>低癌症概率切片数量: {this.state.cancer_number}</Text>
                    <Text style={{ color: '#d4b446' }}>低癌症占比率: <span style={{ fontWeight: 'bold' }}>{(this.state.cancer_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                    <Text style={{ color: "#ee8826" }}>高癌症概率切片数量: {this.state.high_cancer_number}</Text>
                    <Text style={{ color: "#ee8826" }}>高癌症占比率: <span style={{ fontWeight: 'bold' }}>{(this.state.high_cancer_number / this.state.total_num_2 * 100).toFixed(2)}%</span></Text>
                  </div>
                </Descriptions.Item>
                <Descriptions.Item span={4} label="各标签图片占比图">
                  <ReactECharts
                    style={{ width: 300, height: 300, marginLeft: 600 }}
                    option={this.state.pieOption}
                  />
                </Descriptions.Item>
                <Descriptions.Item span={4} label="诊断建议">
                  <div style={{ marginTop: '10px' }}>
                  {this.state.judge == 1 && (
                      <p>预测有较高癌症倾向概率的切片占比最高，由此推断，病人组织趋向于发生癌变，请您注意！</p>
                    )}
                    {this.state.judge == 2 && (
                      <p>预测有癌症倾向概率的切片占比最高，病人组织有发生癌变的可能，请您继续观察。</p>
                    )}
                    {this.state.judge == 3 && (
                      <p>预测无癌症倾向概率的切片占比最高，病人组织发生癌变的可能性较少，请您留意。</p>
                    )}
                     {this.state.judge == 4 && (
                      <p>预测高癌症倾向概率的切片占比大于20%，病人组织趋向于发生癌变，请您注意！</p>
                    )}
                    {this.state.judge == 5 && (
                      <p>预测癌症倾向概率的切片占比大于30%，病人组织有发生癌变的可能，请您继续观察。</p>
                    )}
                  </div>

                </Descriptions.Item>

              </Descriptions>
            </Col>
          </Row>
        </Modal>
        <Modal
          title="未标记切片信息"
          visible={this.state.isModalVisible2}
          onCancel={() => this.setState({ isModalVisible2: false })}
          footer={[
            <Button key="close" onClick={() => this.setState({ isModalVisible2: false })}>关闭</Button>
          ]}
          style={{ top: '50%', transform: 'translateY(-50%)' }} // 将Modal居中显示
        >
          <div style={{ display: 'flex', flexWrap: 'wrap' }}>
            {this.state.do_data.map((imageUrl, index) => (
              <div key={index} style={{ width: '12.5%', marginBottom: '20px', marginRight: '10px', padding: '0 10px' }}>
                <div style={{ marginBottom: '10px' }}>
                  <img src={process.env.REACT_APP_IMAGE_PATH_NEW + "/image_" + this.state.choosePatches[this.state.choosePatches.length - 1] + "/" + imageUrl} style={{ maxWidth: '100%', height: 'auto' }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <Button type="primary" size="small" onClick={(imageUrl) => { this.dowork(this.state.do_data[index]) }}>标记</Button>
                </div>
                <div>{imageUrl}</div>
              </div>
            ))}
            {this.state.do_data.length % 8 !== 0 && (
              // 添加占位元素以保持每行显示 7 张图片
              <div style={{ width: `${(8 - this.state.do_data.length % 8) * 12.5}%` }}></div>
            )}
          </div>
        </Modal>
        <Modal
          title="未标记切片修改页面"
          visible={this.state.isModalVisible3}
          onCancel={() => this.setState({ isModalVisible3: false })}
          footer={[
            <Button key="close" onClick={() => this.setState({ isModalVisible3: false })}>关闭</Button>
          ]}
          style={{ top: '50%', transform: 'translateY(-50%)' }} // 将Modal居中显示
        >
          <Row gutter={[16, 16]}> {/* 使用 gutter 来设置行之间的间隔 */}
            <Col span={12}> {/* 左侧列，占据总宽度的一半 */}
              <img src={process.env.REACT_APP_IMAGE_PATH_NEW + "/image_" + this.state.choosePatches[this.state.choosePatches.length - 1] + "/" + this.state.value1} style={{ maxWidth: '100%', height: 'auto' }} />
              <Text>{this.state.value1}</Text>
            </Col>
            <Col span={12}> {/* 右侧列，占据总宽度的一半 */}
              <Row gutter={[16, 16]}> {/* 使用 gutter 来设置行之间的间隔 */}
                <Col span={24}>
                  <Text>BMM的分数: {this.state.bmm_score}</Text>
                </Col>
                <Col span={24}>
                  <Text>PR的分数: {this.state.spr_score}</Text>
                </Col>
                <Col span={24}>
                  <Button type="primary" color='green' onClick={this.click1}>无癌症 </Button>
                </Col>
                <Col span={24}>
                  <Button type="primary" color='yellow' onClick={this.click2}>有癌症 </Button>
                </Col>
                <Col span={24}>
                  <Button type="primary" color='red' onClick={this.click3}>高癌症 </Button>
                </Col>
              </Row>
            </Col>
          </Row>
        </Modal>
      </>
    );
  }
}

