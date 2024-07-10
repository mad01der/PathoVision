import React, { Component } from "react";
import { Image, Row, Col, Card, Typography, Modal ,Button} from "antd";
import "./MapVision.css"; // Import your custom CSS file
import html2canvas from "html2canvas"; // Import html2canvas library
import { message } from "antd";
const { Text, Title } = Typography;

export default class Pic extends Component {
  constructor(props) {
    super(props);
    this.state = {
      img_len: Object.keys(this.props.WSI_Data.img_id).length,
      modalVisible: false,
      selectedImage: null,
      currentImageIndex: null, // Track the current displayed image index
      choosePatches: props.choosePatches
    };
  }

  handleImageClick = (index) => {
    console.log(this.state.img_len)
    if(this.state.img_len > 1){
       var selectedImage = process.env.REACT_APP_WSI_PATH_NEW + `/${index}.png`;
    }
    else{
       var selectedImage = process.env.REACT_APP_WSI_PATH + `/${index}.png`;
    }
    // const selectedImage = process.env.REACT_APP_WSI_PATH + `/${index}.png`;
    console.log(selectedImage)
    this.setState({ modalVisible: true, selectedImage, currentImageIndex: index });
  };
  
  handleModalClose = () => {
    this.setState({ modalVisible: false, selectedImage: null, zoom: 1 });
  };
  selectPic = {
    click: async (e) => {
      const newTags = this.state.choosePatches.filter(
        (tag) => tag !== e.dataIndex
      );
      if (newTags.length < 10) {
        await newTags.push(e.dataIndex);
        await this.setState({
          choosePatches: newTags,
        });
      } else {
        await message.error("The selected image has reached the limitation!");
      }
      this.props.changeChoosePatches(newTags);
    },
  };
  handleScreenshot = () => {
    const container = document.querySelector(".image-container");

    // Only capture the currently displayed image
    const selectedImage = container.querySelector(`#image-${this.state.currentImageIndex}`);

    html2canvas(selectedImage).then((canvas) => {
      // Convert canvas to image
      const screenshot = canvas.toDataURL("image/png");

      // Create a link element to download the screenshot
      const link = document.createElement("a");
      link.download = "screenshot.png";
      link.href = screenshot;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  };

  render() {
    const { img_len, modalVisible, selectedImage, zoom, currentImageIndex } = this.state;

    const imagePaths = Array.from({ length: img_len }, (_, index) => ({
      src: img_len > 1  ? process.env.REACT_APP_WSI_PATH_NEW + `/${index}.png` : process.env.REACT_APP_WSI_PATH + `/${index}.png`,
      alt: `Image ${index}`,
    }));

    return (
      <Card bordered={false} hoverable={true}>
        <Col span={15}>
          <Title level={4}> WSI 图片列表 </Title>
        </Col>
        <Col>
          <Text>点击查看细节</Text>
        </Col>
        <Row>
          <Col span={24}>
            <div className="image-container">
              {imagePaths.map((image, index) => (
                <div key={index} className="grid-img-container">
                <Row>
                  <div onClick={() => this.handleImageClick(index)}>
                    <div className="enlarged-image-container">
                      <Image
                        id={`image-${index}`}
                        className="grid-img"
                        preview={false}
                        src={image.src}
                        alt={image.alt}
                        style={{ cursor: "pointer" }}
                      />
                    </div>
                    
                    <Col>
                     <Text> WSI-{index + 1}</Text>
                    </Col>
                  </div>
                  <Button onClick={() => this.selectPic.click({ dataIndex: index })}>选择</Button>
                </Row>
              </div>              
              ))}
            </div>
          </Col>
        </Row>
        {modalVisible && (
          <Modal visible={modalVisible} onCancel={this.handleModalClose} footer={null} width={900}>
            {selectedImage && (
              <>
                <Image
                  src={selectedImage}
                  alt="Enlarged Image"
                  style={{ width: `100%`, height: "auto", transform: `scale(${zoom})` }}
                />
                <div style={{ textAlign: "center", marginTop: "10px" }}>
                  <button className="action-button download-button" onClick={this.handleScreenshot}>Download Picture</button>
                </div>
              </>
            )}
          </Modal>
        )}
      </Card>
    );
  }
}
